import torch.nn as nn
import tensorly as tl
import torch
import torch.nn.functional as F
from typing import Dict
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import tensor_train
from functools import partial
from transformers.modeling_outputs import BaseModelOutput

tl.set_backend('pytorch')

def tensorized_multiplication_experts(self, X, tt_cores, m_factors, n_factors, gates):
    """
    - X: [B, m]
    - tt_cores: list of (len(m_factors)+len(n_factors)) cores,
                each core has shape [E, r_i, factor_dim, r_{i+1}],
                where E is the number of experts.
    - gates: [B, E], gating weights for each sample.
      e.g. from a router or gumbel_softmax.
    - returns: [B, prod(n_factors)]
    """
    B = X.shape[0]
    seq_len = X.shape[1]
    # (1) Reshape X => e.g. [B, m_factors[::-1]] + unsqueeze rank
    shape_x = [B] + [seq_len]+  m_factors[::-1]
    # print("\nInput shape reshaped to match the m_factors in reverse", shape_x)
    tt_state = X.view(shape_x).unsqueeze(1)  # => [B, 1, ...]
    # print("\nShape of tt_state after unsqueeze to add a space for r", tt_state.shape)
    tt_state.to(self.device)

    num_m = len(m_factors)
    num_n = len(n_factors)
    total_cores = num_m + num_n

    if len(tt_cores) != total_cores:
        raise ValueError(f'Expected {total_cores} TT-cores, got {len(tt_cores)}')

    # We'll do the same flow: contract out input factors, then add output factors.
    # But each time, we must "mask" the stacked core by gates, sum over E.

    for i in range(num_m):
        # (a) core: [E, r_i, m_i, r_{i+1}]
        core = tt_cores[i].to(self.device)

        # (b) Apply gating => shape [B, r_i, m_i, r_{i+1}]
        #    gates => [B, E], expand => [B, E, 1, 1, 1]
        #    core => [E, r_i, m_i, r_{i+1}] => unsqueeze(0) => [1, E, r_i, m_i, r_{i+1}]
        #    multiply & sum out E => [B, r_i, m_i, r_{i+1}]
        gates_exp = gates.view(B,-1,1, 1, 1).to(self.device)
        # if i == 0:
        #     print("\nGates expanded shape", gates_exp.shape)
        #     print("\nCore shape with experts", core.shape)
        core_expanded = core.unsqueeze(0).to(self.device)  # [1, 1, E, r_i, m_i, r_{i+1}]
        # if i == 0:
        #     print("\nCore expanded shape with experts to match gate and select cores for all the inputs", core_expanded.shape)
        masked_core_m = (core_expanded * gates_exp).sum(dim=1).to(self.device)
        # if not hasattr(self, 'print_count1'):
        #     print("\n","*"*50)
        #     print("gates", gates[0:5])
        #     print(f"\nMasked Core for core {i}", masked_core_m[0:5])
        #     self.print_count1 = True
        # if i == 0:
        #     print("\nMasked Core shape", masked_core.shape)
        # => [B, r_i, m_i, r_{i+1}]

        # (c) einsum with tt_state
        # tt_state => [B, r_i, ..., m] (the last dimension is the factor dim if i < num_m)
        eq_in = "b r ... m, b r m p -> b p ..."
        tt_state = torch.einsum(eq_in, tt_state, masked_core_m).to(self.device)
        # if i==0:
        #     print("\nShape of tt_state after einsum with masked core to reduce input_dim", tt_state.shape)
        # if i==len(m_factors)-1:
        #     print("\nFinal Shape of tt_state after einsumth masked core to reduce input_dim", tt_state.shape)
        # => [B, r_{i+1}, leftover...] wi

    # Now do output factors (which add a dimension).
    start_n = num_m
    for i in range(num_n):
        core = tt_cores[start_n + i].to(self.device)  # shape [E, r_i, n_i, r_{i+1}]
        # Mask by gating
        gates_exp = gates.view(B, -1, 1, 1, 1).to(self.device)
        core_expanded = core.unsqueeze(0).to(self.device)  # => [1, E, r_i, n_i, r_{i+1}]
        masked_core_n = (core_expanded * gates_exp).sum(dim=1).to(self.device)
        # if not hasattr(self, 'print_count'):
        #     print("\n","*"*50)
        #     print("i", i)
        #     print("gates", gates[0:5])
        #     print(f"\nMasked Core for core {i+start_n}", masked_core_n[0:5])
        #     self.print_count = True
        # => [B, r_i, n_i, r_{i+1}]

        # eq_out: "b r ..., b r n p -> b p ... n"
        eq_out = "b r ..., b r n p -> b p ... n"
        tt_state = torch.einsum(eq_out, tt_state, masked_core_n).to(self.device)
        # if i==0:
        #     print("\nShape of tt_state after einsum with masked core to add output_dim", tt_state.shape)
        # if i==len(n_factors)-1:
        #     print("\nFinal Shape of tt_state after einsum with masked core to add output_dim", tt_state.shape)

    # Flatten
    Z = tt_state.view(B, seq_len, -1).to(self.device)
    # print("\nFinal Shape of output after flattening", Z.shape)
    return Z

class MultiStepRouter(nn.Module):
    def __init__(self, input_dim, num_experts, router_type, num_heads=4, hidden_dim=1024):
        super(MultiStepRouter, self).__init__()
        self.m = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.router_type = router_type

        self.expand = nn.Linear(self.m, 1024, bias=True)  # Expand from m to 1024
        # self.hidden = nn.Linear(1024, 1024, bias=True)  # Hidden layer

        if router_type == "attention":
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                              num_heads=num_heads, 
                                              batch_first=True)
            
        self.project = nn.Linear(1024, num_experts, bias=True)  # Project 1024 to num_experts
    
    def forward(self, x):
        x = self.expand(x)  # Expand to 1024
        # x = torch.relu(x)    # Apply non-linearity
        # x= self.hidden(x)  # Hidden layer
        x = F.relu(x)
        if self.router_type == "attention":
            attn_output, _ = self.attn(query=x, key=x, value=x)
            logits = self.project(attn_output)
            return logits
        
        elif self.router_type == "multi_layer":
            x = self.project(x)  # Project down to num_experts
            return x
        
        else:
            raise ValueError("Invalid router type")

class MoEsparseRouting(nn.Module):
    def __init__(self, 
                 base_module: nn.Module, 
                 config: Dict,
                 ):
        super().__init__()                      
        self.base_module = base_module
        self.m_factors_q = config["m_factors_q"]
        self.n_factors_q = config["n_factors_q"]
        self.m_factors_v = config["m_factors_v"]
        self.n_factors_v = config["n_factors_v"]

        self.experts = config["experts_dict"]
        self.common_alpha = config["common_alpha"]
        self.num_experts = len(self.experts)
        self.device = config["device"]
        self.router_type = config["router_type"]
        self.gumbel_temperature = config["gumbel_temperature"]
        self.model_name = config["model_name"]
        self.gates = None
        self.num_cores_q = len(config["qshape"])
        self.num_cores_v = len(config["vshape"])

        # Compute m, n for Query
        self.m_q = 1
        for f in self.m_factors_q:
            self.m_q *= f
        self.n_q = 1
        for f in self.n_factors_q:
            self.n_q *= f
        
        # Compute m, n for Value
        self.m_v = 1
        for f in self.m_factors_v:
            self.m_v *= f
        self.n_v = 1
        for f in self.n_factors_v:
            self.n_v *= f

        # A trainable router => from [B, m] => [B, E]
        if self.router_type == "single_layer":
            self.router = nn.Linear(self.m_q, self.num_experts, bias=True)
            print("Router type is single layer")
        elif self.router_type == "multi_layer" or "attention":
            self.router = MultiStepRouter(input_dim=self.m_q, 
                                          num_experts=self.num_experts, 
                                          router_type=self.router_type)
            print("Router type is multi layer")
 
    def custom_query_forward(self, 
                             X, 
                             base_layer_weight, 
                             base_layer_bias, 
                             gates, 
                             stacked_query_cores,
                             *args, **kwargs):
        # print("*"*50)
        # # (c) Store as buffer
        tt_cores_stacked = stacked_query_cores
        # print(tt_cores_stacked[5])  #should print ttlora_core_5 for all experts in order
        base_layer_out = F.linear(input=X, weight=base_layer_weight, bias=base_layer_bias)
        
        # print(f"Printing passed stacked_query_cores to custom_query_forward")
        # print("*"*50)
        # for i, core_list in enumerate(tt_cores_stacked):
        #     print(f"core{i}:", core_list)
        #     break
        # Count the total TT-cores
        self.num_m = len(self.m_factors_q)
        self.num_n = len(self.n_factors_q)
        self.num_cores = self.num_m + self.num_n
        if len(tt_cores_stacked) != self.num_cores:
            raise ValueError(f'Expected {self.num_cores} cores, got {len(tt_cores_stacked)}')

        # Figure out E from the shape of the first core
        example_core = tt_cores_stacked[0]
        self.num_experts = example_core.shape[0]  # first dimension => E

        # (1) Store TT-cores as buffers => freeze them
        # if self.train_router_only:
        #     for i, core in enumerate(tt_cores_stacked):
        #         # shape => [E, r_i, factor_dim, r_{i+1}]
        #         self.register_buffer(f"core_{i}", core)
        # else:
        #     # If you wanted them trainable, store as parameters
        #     # But here the focus is on freezing.
        #     raise NotImplementedError("We only do freezing in this example")
        
        # (d) Collect the stacked TT-cores from buffers
        # We'll build a python list in the correct order
        stacked_cores_list = tt_cores_stacked
        # print(stacked_cores_list[3])
        # for i in range(self.num_cores):
        #     c_buf = getattr(self, f"core_{i}")  
        #     # shape => [E, r_i, factor_dim, r_{i+1}]
        #     stacked_cores_list.append(c_buf)

        # print("\nInside custom query forwrad\n", gates[0:5])
        # (e) Call the helper function to compute input with tensor cores => [B, s_l, n]
        ttlora_x_computation = tensorized_multiplication_experts(self, 
                                                                 X, 
                                                                 stacked_cores_list, 
                                                                 self.m_factors_q, 
                                                                 self.n_factors_q, 
                                                                 gates)
        
        # (f) Override the forward function of the query layer
        alpha = self.common_alpha
        out = ttlora_x_computation*alpha

        # scaling_factor = torch.mean(Q) + torch.mean(X)
        return out + base_layer_out

    def custom_value_forward(self, 
                             X, 
                             base_layer_weight, 
                             base_layer_bias, 
                             gates, 
                             stacked_value_cores,
                             *args, **kwargs):
        tt_cores_stacked = stacked_value_cores
        base_layer_out = F.linear(input=X, weight=base_layer_weight, bias=base_layer_bias)

        # Count the total TT-cores
        self.num_m = len(self.m_factors_v)
        self.num_n = len(self.n_factors_v)
        self.num_cores = self.num_m + self.num_n
        if len(tt_cores_stacked) != self.num_cores:
            raise ValueError(f'Expected {self.num_cores} cores, got {len(tt_cores_stacked)}')

        # Figure out E from the shape of the first core
        example_core = tt_cores_stacked[0]
        self.num_experts = example_core.shape[0]  # first dimension => E

        ttlora_x_computation = tensorized_multiplication_experts(self, 
                                                                 X, 
                                                                 tt_cores_stacked, 
                                                                 self.m_factors_v, 
                                                                 self.n_factors_v, 
                                                                 gates)
        
        alpha = self.common_alpha
        out = ttlora_x_computation*alpha
        return out + base_layer_out
    
    def count_labels_of_experts_selection(self, gates):
        expert_key = list(self.experts.keys())
        counts = {key: 0 for key in expert_key}
        for row in gates:
            idx = torch.argmax(row)
            counts[expert_key[idx]] += 1
            
        counts["total"] = sum(counts.values())
        for expert in expert_key:
            counts[expert] = (counts[expert]/counts["total"])*100
        return counts

    def forward(self, X, *args, **kwargs):
        X_temp = X
        X= X.to(self.device)
        gumbel_temperature = self.gumbel_temperature
        if "roberta" in self.model_name:
            B = X.size(0)
            
            pooled_hidden_states = X.float().mean(dim=1)
            
            logits = self.router(pooled_hidden_states)
            print("\nexpert keys", self.experts.keys())
            print("\nlogits", logits[:5])
            self.gates = F.gumbel_softmax(logits, tau=gumbel_temperature, hard=True).to(self.device)
            print("\ngates", self.gates[:5])
            print(f'\n Selected Experts of this Batch size {B}\n', 
                self.count_labels_of_experts_selection(self.gates))
            layer_idx = 0
            for layer in self.base_module.layer:
                # print("\ninside layer", layer_idx)

                # Start Collecting the TT-cores from here, for this layer, for all experts
                # access_first_expert = next(iter(self.experts.values()))
                ##################################################For query######################################
                # (a) Collect query TT-cores for all experts
                list_query_cores = [[] for _ in range(self.num_cores_q)]
                # list_query_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["query"]))]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["query"].items()):
                        list_query_cores[i].append(tensor)
                
                # print("\n Expert stacking for mrpc", list_query_cores[0][1])

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_query_cores = [torch.stack(core_list) for core_list in list_query_cores]
                # # Check by printing the stacked cores       
                # print(f"Printing stacked_query_cores only for layer_{layer_idx} and batch size of {X.shape[0]}")
                # print("*"*50)
                # for i, core_list in enumerate(stacked_query_cores):
                #     print(f"core{i}:", core_list)
                #     break
                layer.attention.self.query.forward = partial(self.custom_query_forward, 
                                                            base_layer_weight=layer.attention.self.query.weight, 
                                                            base_layer_bias= layer.attention.self.query.bias,
                                                            gates=self.gates, 
                                                            stacked_query_cores=stacked_query_cores)            
                ##################################################For Value######################################
                # (a) Collect query TT-cores for all experts
                list_value_cores = [[] for _ in range(self.num_cores_v)]
                # list_value_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["value"]))]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["value"].items()):
                        list_value_cores[i].append(tensor)

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_value_cores = [torch.stack(core_list) for core_list in list_value_cores]
                # print(stacked_value_cores[7])
                # Check by printing the stacked cores
                # if layer_idx == 0 and not hasattr(self, 'printed_value_cores'):
                #     print(f"Printing stacked_value_cores only for layer_{layer_idx} and Batch 0")
                #     print("*"*50)
                #     for i, core_list in enumerate(stacked_query_cores):
                #         print(f"core{i}:", core_list.shape)
                #     self.printed_value_cores = True
                layer.attention.self.value.forward = partial(self.custom_value_forward, 
                                                            base_layer_weight=layer.attention.self.value.weight,
                                                            base_layer_bias= layer.attention.self.value.bias,
                                                            gates=self.gates, 
                                                            stacked_value_cores=stacked_value_cores
                                                            )            
                
                #increase the layer index
                layer_idx += 1
            
            # Forward through transformer layers
            return self.base_module(X, *args, **kwargs)
        
        elif "llama" in self.model_name:
            
            # print("\nType of X in forward of MoESparse", X.dtype)
            # print("\nshape of X before embed_tokens",X.shape)
            X_temp = self.base_module.embed_tokens(X_temp).to(self.device)
            # print("type of embed_token layer ",self.base_module.embed_tokens.weight.shape)
            # print("\nshape of X after embed_tokens",X_temp.shape)
            B = X_temp.size(0)
            pooled_hidden_states = X_temp.float().mean(dim=1)
            print("\nexpert keys", self.experts.keys())

            # print("\nshape of pooled_hidden_states",pooled_hidden_states.shape)
            logits = self.router(pooled_hidden_states)
            print("\nshape of gate logits",logits.shape)
            self.gates = F.gumbel_softmax(logits, tau=gumbel_temperature, hard=True).to(self.device)
            print("\nshape of gates",self.gates.shape)
            print(f'\nSelected Experts of this Batch size {B}\n', 
                self.count_labels_of_experts_selection(self.gates))
            layer_idx = 0
            for layer in self.base_module.layers:
                ##################################################For query######################################
                # (a) Collect query TT-cores for all experts
                list_query_cores = [[] for _ in range(self.num_cores_q)]
                # list_query_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["query"]))]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["query"].items()):
                        list_query_cores[i].append(tensor)
                
                # print("\n Expert stacking for mrpc", list_query_cores[0][1])

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_query_cores = [torch.stack(core_list) for core_list in list_query_cores]
                layer.self_attn.q_proj.forward = partial(self.custom_query_forward, 
                                                            base_layer_weight=layer.self_attn.q_proj.weight,
                                                            base_layer_bias = None, 
                                                            gates=self.gates, 
                                                            stacked_query_cores=stacked_query_cores)            
                ##################################################For Value######################################
                # (a) Collect query TT-cores for all experts
                list_value_cores = [[] for _ in range(self.num_cores_v)]
                # list_value_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["value"]))]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["value"].items()):
                        list_value_cores[i].append(tensor)

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_value_cores = [torch.stack(core_list) for core_list in list_value_cores]
                layer.self_attn.v_proj.forward = partial(self.custom_value_forward, 
                                                            base_layer_weight=layer.self_attn.v_proj.weight,
                                                            base_layer_bias= None,
                                                            gates=self.gates, 
                                                            stacked_value_cores=stacked_value_cores)            
                
                #increase the layer index
                layer_idx += 1
            
            # Forward through transformer layers
            # print("type of X before passing to base_module", X.dtype, X.shape)
            model_output = self.base_module(X, *args, **kwargs)
            # print(dir(model_output))
            # Check if 'last_hidden_state' exists and print its shape

            return model_output

class MoEsparseRoutingForClassification(nn.Module):
    ####################################For Classification Head#########################################
    def __init__(self, 
                 base_classifier: nn.Module, 
                 config: dict,
                 router: nn.Module,
                 ):
        super().__init__()
        self.base_classifier = base_classifier
        self.experts = config["experts_dict"]
        self.device = config["device"]
        self.broadcast_gates = router
        self.num_experts = len(self.experts)
        self.model_name = config["model_name"]
        # print("Inside Classification Initialization")

    
    def custom_classifier_dense_forward(self, X,
                                        selected_weights:torch.Tensor,
                                        selected_biases,
                                        *args, **kwargs):
        X= X.unsqueeze(1)
        if selected_biases is None:
            out = (X * selected_weights).sum(2)
        else:
            out = (X * selected_weights).sum(2) + selected_biases
        # print("output shape", out.shape)
        return out
    
    def custom_classifier_out_proj_forward(self, X,
                                        selected_weights:torch.Tensor,
                                        selected_biases,
                                        *args, **kwargs):
        
        X= X.unsqueeze(1)
        out = (X * selected_weights).sum(2) + selected_biases
        # print("\noutput shape from custom out_proj_forward", out.shape)
        return out
    
    def custom_score_forward(self, X,
                             selected_weights:torch.Tensor,
                                        selected_biases,
                                        *args, **kwargs):
        # print("\nshape of X inside classifier projection layer", X.shape)
        # print("\nshape of selected_weights", selected_weights.shape)
        # # X = X.mean(dim=1)
        # print("\nshape of X after mean", X.shape)
        # X= X.unsqueeze(2)
        # weights = selected_weights.unsqueeze(1)
        # print("\nshape of X after unsqueeze", X.shape)
        # out = (X * selected_weights).sum(3)
        out = torch.einsum('bsl,bod->bso', X, selected_weights)
        # print("\nshape of out after sum", out.shape)
        # sys.exit(1)
        return out

    def forward(self, X, *args, **kwargs):
        # print("Inside MoEsparseRoutingForClassification Forward")

        if "roberta" in self.model_name:
            # Stack classifier dense weights and biases for all experts
            stacked_dense_weights = torch.stack([self.experts[expert]["classifier"]["dense"]["weight"] for expert in self.experts]).to(self.device)
            stacked_dense_biases = torch.stack([self.experts[expert]["classifier"]["dense"]["bias"] for expert in self.experts]).to(self.device)
            # Stack out_proj weights and biases for all experts
            stacked_out_proj_weights = torch.stack([self.experts[expert]["classifier"]["out_proj"]["weight"] for expert in self.experts]).to(self.device)
            stacked_out_proj_biases = torch.stack([self.experts[expert]["classifier"]["out_proj"]["bias"] for expert in self.experts]).to(self.device)

            gates = self.broadcast_gates.gates.to(self.device)
            B, seq_len, _ = X.shape

            # **Expand gates for proper broadcasting**
            gates_dense = gates.view(B, -1, 1, 1).to(self.device)  # Shape: [B, num_experts, 1, 1] for dense weights
            gates_out_proj = gates.view(B, -1, 1).to(self.device)  # Shape: [B, num_experts, 1] for out_proj biases

            if gates is None:
                raise ValueError("Gates have not been computed. Ensure the encoder's forward pass runs before classification.")

            # Expand stacked weights to match batch size
            stacked_dense_weights = stacked_dense_weights.unsqueeze(0).to(self.device)  # Shape: [1, 4, 768, 768]
            stacked_dense_biases = stacked_dense_biases.unsqueeze(0).to(self.device)  # Shape: [1, 4, 768]

            stacked_out_proj_weights = stacked_out_proj_weights.unsqueeze(0).to(self.device)  # Shape: [1, 4, 2, 768]
            stacked_out_proj_biases = stacked_out_proj_biases.unsqueeze(0).to(self.device)  # Shape: [1, 4, 2]

            # **Compute dynamically selected classifier weights**
            selected_dense_weights = (stacked_dense_weights * gates_dense).sum(dim=1).to(self.device)  # Shape: [B, 768, 768]
            selected_dense_biases = (stacked_dense_biases * gates_out_proj).sum(dim=1).to(self.device)  # Shape: [B, 768]
            # print("dtype of selected_dense_biases", type(selected_dense_biases))
            # print("biases", selected_dense_biases)
            selected_out_proj_weights = (stacked_out_proj_weights * gates_out_proj.unsqueeze(-1)).sum(dim=1).to(self.device)  # Shape: [B, 2, 768]
            selected_out_proj_biases = (stacked_out_proj_biases * gates_out_proj).sum(dim=1).to(self.device)  # Shape: [B, 2]

            self.base_classifier.dense.forward = partial(self.custom_classifier_dense_forward,
                                                    selected_weights=selected_dense_weights, 
                                                    selected_biases=selected_dense_biases)

            self.base_classifier.out_proj.forward = partial(self.custom_classifier_out_proj_forward, 
                                                    selected_weights=selected_out_proj_weights,
                                                    selected_biases=selected_out_proj_biases)

            classifier_out = self.base_classifier(X, *args, **kwargs)
            # print("classifier_out of roberta", classifier_out)
            # sys.exit(1)
            return classifier_out

        elif "llama" in self.model_name:
            # print("X inside classifier", X.shape, *args, **kwargs)
            # Stack classifier dense weights and biases for all experts
            stacked_score_weights = torch.stack([self.experts[expert]["score"]["weight"] for expert in self.experts]).to(self.device)
            # print("stacked_score_weights", stacked_score_weights.shape)

            gates = self.broadcast_gates.gates.to(self.device)
            # print("gates", gates.shape)

            B, seq_len, _ = X.shape
            # print("batch_size inside classification class", B)

            # **Expand gates for proper broadcasting**
            gates_score = gates.view(B, -1, 1).to(self.device)  # Shape: [B, 4, 1] for score
            # print("gates_score after expansion", gates_score.shape)
            if gates is None:
                raise ValueError("Gates have not been computed. Ensure the encoder's forward pass runs before classification.")

            # Expand stacked weights to match batch size
            stacked_score_weights = stacked_score_weights.unsqueeze(0).to(self.device)  # Shape: [1, 4, 2, 4096]
            # print("stacked_score_weights after expansion", stacked_score_weights.shape)

            # **Compute dynamically selected classifier weights**
            selected_score_weights = (stacked_score_weights * gates_score.unsqueeze(-1)).sum(dim=1).to(self.device)  # Shape: [B, 2, 4096]
            # print("selected_score_weights after gate multiplication", selected_score_weights.shape)
            self.base_classifier.forward = partial(self.custom_score_forward, 
                                                    selected_weights=selected_score_weights,
                                                    selected_biases=None)
            classifier_out = self.base_classifier(X, *args, **kwargs)
            # print("classifier_out of llama", classifier_out)
            # sys.exit(1)
            return classifier_out
