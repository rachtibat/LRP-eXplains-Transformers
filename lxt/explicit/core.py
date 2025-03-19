import torch.nn as nn
from lxt.explicit.rules import WrapModule
from lxt.explicit.modules import INIT_MODULE_MAPPING
from lxt.explicit.check import WHITELIST, BLACKLIST, SYMBOLS
from transformers.utils.fx import HFTracer, get_concrete_args
from torch.fx import GraphModule
from warnings import warn
from tabulate import tabulate

class Composite:
    """
    Base class for composites. A composite is a collection of rules that are applied to a model.

    Parameters
    ----------
    layer_map: dict
        A dictionary of the form {layer_type: rule} where 'layer_type' is a torch.nn.Module or a string corresponding to the module name or a function (as traced by torch.fx),
        and 'rule' is a LXT.rules.WrapModule subclass or a torch.nn.Module or a torch.autograd.Function.
    canonizers: list
        A list of canonizers to apply to the model before applying the rules.
    zennit_composite: ZennitComposite
        An optional ZennitComposite to apply to the model after applying the rules.
    """

    def __init__(self, layer_map, canonizers=[], zennit_composite=None) -> None:
        
        self.layer_map = layer_map
        self.original_modules = []

        self.module_summary, self.function_summary = {}, {}

        self.canonizers = canonizers
        self.canonizer_instances = []

        for c in canonizers:
            if isinstance(c, type):
                raise ValueError(f"You must call the canonizer {c}(). You passed the class instead of an instance.")


        self.zennit_composite = zennit_composite

    def register(self, parent: nn.Module, dummy_inputs: dict=None, tracer=HFTracer, verbose=False, no_grad=True) -> None:
        """
        Register the composite to the model and apply the canonizers to the model, if available.

        Parameters
        ----------
        parent: torch.nn.Module
            The parent model (origin) to which the composite should be registered
        dummy_inputs: dict
            A dictionary of dummy inputs to trace the model with torch.fx. The keys are the names of the inputs and the values are the dummy tensors.
            This is only necessary if the composite contains function rules.
        verbose: bool
            print if a module got a rule or not for debugging
        no_grad: bool
            Turn off gradient computation for the parameters by setting the requires_grad to False. These gradients are not needed and can save memory.
        """

        if no_grad:
            for param in parent.parameters():
                param.requires_grad = False

        for canonizer in self.canonizers:

            try:
                # LXT canonizers
                instances = canonizer.apply(parent, verbose)
            except TypeError:
                # zennit canonizers dont have 'verbose' argument
                instances = canonizer.apply(parent)

            self.canonizer_instances.extend(instances)

        # first, attach the rules to the modules
        # then, attach the rules to the functions
        module_map, fn_map = self._parse_rules(self.layer_map)
        if module_map:
            self._iterate_children(parent, module_map)

        if fn_map or dummy_inputs:
            parent = self._iterate_graph(parent, dummy_inputs, fn_map, module_map, tracer)
        
        # register an optional zennit composite
        if self.zennit_composite:
            if verbose:
                print("-> register ZENNIT composite", self.zennit_composite)
            self.zennit_composite.register(parent)

        if verbose and (fn_map or dummy_inputs):
            self.print_summary()

        return parent

    def _parse_rules(self, layer_map):

        module_map, fn_map = {}, {}
        for key, value in layer_map.items():

            if isinstance(key, str) or isinstance(key, nn.Module.__class__): #TODO. wrap module?
                module_map.update({key: value})
            elif callable(key):
                fn_map.update({key: value})
            else:
                raise ValueError(f"Key {key} must be a subclass of nn.Module, a string or a callable function.")
            
        return module_map, fn_map


    def _iterate_children(self, parent: nn.Module, rule_dict):
        """
        Recursive function to iterate through the children of a module and attach the rules to the modules.
        """

        for name, child in parent.named_children():

            child = self._attach_module_rule(child, parent, name, rule_dict)
            self._iterate_children(child, rule_dict)

    def _attach_module_rule(self, child, parent, name, rule_dict):
        """
        Attach a rule to the child module if it either 1. has a name equal to 'layer_type' or 2. is an instance of the 'layer_type' (see rule_dict).
        In this case, if the rule is a subclass of WrapModule, the module is wrapped with the rule and attached to the parent as an attribute.
        If the rule is a torch.nn.Module, the module is directly replaced with the rule by copying the parameters and then attached to the parent as an attribute.
        """

        for layer_type, rule in rule_dict.items():
            
            # check if the layer_type is a string and or if the layer_type is a class and the child is an instance of it
            if (isinstance(layer_type, str) and layer_type == child) or isinstance(child, layer_type):
                
                if issubclass(rule, WrapModule):
                    # replace module with LXT.rules.WrapModule and attach it to parent as attribute
                    xai_module = rule(child)
                elif issubclass(rule, nn.Module):
                    # replace module with LXT.module and attach it to parent as attribute
                    # INIT_MODULE_MAPPING contains the correct function for initializing and copying the parameters and buffers
                    xai_module = INIT_MODULE_MAPPING[rule](child, rule)
                    # return the new module to iterate through its children to attach the rules
                    child = xai_module
                else:
                    raise ValueError(f"Rule {rule} must be a subclass of WrapModule or a torch.nn.Module")

                setattr(parent, name, xai_module)

                # save original module to revert the composite in self.remove()
                self.original_modules.append((parent, name, child))

                return child
        
        # could not find a rule for the module

        return child


    def _iterate_graph(self, model, dummy_inputs, fn_map, module_map, tracer=HFTracer):

        #TODO: dont trace through detach

        assert isinstance(dummy_inputs, dict), "dummy_inputs must be a dictionary"
        assert dummy_inputs, "dummy_inputs must not be empty"

        graph = tracer().trace(model, concrete_args=get_concrete_args(model, dummy_inputs.keys()), dummy_inputs=dummy_inputs)

        module_types = list(module_map.values())
        for node in graph.nodes:

            self._attach_function_rule(node, fn_map, module_types)

        graph.lint()
        traced = GraphModule(model, graph)
        traced.recompile()

        return traced


    def _attach_function_rule(self, node, fn_map, module_types):
        """
        Attach a rule to a function in the graph if the function is in the composite.
        For now, we only replace functions, not methods (e.g. tensor.add, tensor.sum, etc.).

        Parameters
        ----------
        node: torch.fx.Node
            A node in the graph.
        fn_map: dict
            A dictionary of the form {function: rule} where 'function' is a callable function and 'rule' is a callable function.
        module_types: list
            A list of module types that have already been wrapped by a LRP rule.
        
        """

        # if the parent module of the node has already been wrapped by a LRP rule, we do not want to replace a function inside it
        if self._check_already_wrapped(node, module_types):
            
            # record all modules that have already been wrapped by a LRP rule as replaced for verbose debugging
            self._add_to_module_summary(node, True)

            return False

        if node.op == 'call_function':
            
            # if the function is in the composite, replace it with the LRP function
            if node.target in fn_map:

                # record all explained functions for verbose debugging
                self._add_to_fn_summary(node, True)

                node.target = fn_map[node.target]

                return True

            # record all missing functions for verbose debugging
            self._add_to_fn_summary(node, False)

        elif node.op == 'call_method':
            
            # we limit ourselves for now to replace only functions, not methods (e.g. tensor.add, tensor.sum, etc.)
            # record all methods for verbose debugging as not replaced
            self._add_to_fn_summary(node, False)

        elif node.op == 'call_module':
            
            # record all modules not wrapped by a LRP rule as not replaced for verbose debugging
            self._add_to_module_summary(node, False)

        
        return False
    

    def _check_already_wrapped(self, node, module_types):
        """
        Check if a module has already been wrapped with a rule by reading the module stack information 
        in the node meta data. This is important because, we do not want to replace a function inside 
        a module that has already been replaced or wrapped by a LXT rule.

        Parameters
        ----------
        node: torch.fx.Node
            A node in the graph.
        module_types: list
            A list of module types that have already been wrapped by a LRP rule.
        """

        if "nn_module_stack" in node.meta:
            for l_name, l_type in node.meta["nn_module_stack"].items():
                if l_type in module_types:
                    return True
            return False
        else:
            return False
        
    def _add_to_module_summary(self, node, replaced: bool):
        """
        This method is used for verbose debugging. It records all modules that have been replaced or not replaced by the composite.
        If a module is replaced by the user in the composite, we record it as "True". If a module is not replaced, we record it as "False".

        Parameters
        ----------
        module: torch.nn.Module
            A module in the graph.
        replaced: bool
            A boolean indicating if the module has been replaced by the user in the composite.
        """
        
        l_name, l_type = list(node.meta["nn_module_stack"].items())[-1]

        if l_type not in self.module_summary:
            self.module_summary[l_type] = replaced
    
    
    def _add_to_fn_summary(self, node, replaced: bool):
        """
        This method is used for verbose debugging. It records all functions that have been replaced or not replaced by the composite.
        We provide a non-exhaustive whitelist of functions that are compatible with LRP (i.e. their gradients are equal to LRP relevances) and
        a blacklist of functions that are not compatible with LRP. If a function is in the whitelist or replaced by the user in the composite, we record it as "True". 
        If a function is in the blacklist, we record it as "False". If a function is not in the whitelist or the blacklist, we record it as "unknown".

        Parameters
        ----------
        node: torch.fx.Node
            A node in the graph.
        replaced: bool
            A boolean indicating if the function has been replaced by the user in the composite.
        """
        
        # get the parent module name where the function is located
        if "nn_module_stack" in node.meta:
            module_name = list(node.meta["nn_module_stack"].values())[-1]
        else:
            module_name = "Root"

        if module_name not in self.function_summary:
            self.function_summary[module_name] = {}

        if replaced:
            self.function_summary[module_name][node.target] = "replaced"
        elif node.target in WHITELIST:
            self.function_summary[module_name][node.target] = "compatible"
        elif node.target in BLACKLIST:
            self.function_summary[module_name][node.target] = "problematic"
        else:
            self.function_summary[module_name][node.target] = "unknown"


    def print_summary(self):

        headers = ["Parent Module", "Function", "Replaced", "LRP compatible"]

        data = []
        for module in self.module_summary:
            if self.module_summary[module]:
                replaced = SYMBOLS["true"]
                compatible = "-"
            else:
                replaced = "-"
                compatible = SYMBOLS["unknown"]
            data.append([module, "-", replaced, compatible])

        for module, functions in self.function_summary.items():
            for function, rating in functions.items():
                if rating == "replaced":
                    replaced = SYMBOLS["true"]
                    compatible = SYMBOLS["true"]
                elif rating == "compatible":
                    replaced = "-"
                    compatible = SYMBOLS["true"]
                elif rating == "problematic":
                    replaced = "-"
                    compatible = SYMBOLS["false"]
                else:
                    replaced = "-"
                    compatible = SYMBOLS["unknown"]
                data.append([module, function, replaced, compatible])

        table = tabulate(data, headers=headers, tablefmt="grid")
        print(table)


    def remove(self):
        """
        Remove the composite from the model and revert the original modules.
        #TODO: in-depth explanation
        """

        warn("This functionality is not yet fully tested. Please check the model after removing the composite.")

        if self.function_summary:
            warn("Some functions have been replaced by tracing the model with torch.fx. You can't reverse function replacements, but only nn.Module and Zennit replacements.")

        for parent, name, module in self.original_modules:
            rule = getattr(parent, name)
            setattr(parent, name, module)
            del rule

        for instance in self.canonizer_instances:
            instance.remove()

        self.original_modules = []
        self.canonizer_instances = []
        
        if self.zennit_composite is not None:
            self.zennit_composite.remove()

    def context(self, module, verbose=False):
       
        return CompositeContext(module, self, verbose)
    


class CompositeContext:
    '''
    A context object to register a composite in a context and remove the associated hooks and canonizers afterwards.
    Taken from the 'zennit' library for neural network interpretability.

    Parameters
    ----------
    module: torch.nn.Module
        The module to which composite should be registered.
    composite: Composite
        The composite which shall be registered to module.
    '''
    def __init__(self, module, composite, verbose):
        self.module = module
        self.composite = composite
        self.verbose = verbose

    def __enter__(self):
        self.composite.register(self.module, self.verbose)
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.composite.remove()
        return False


