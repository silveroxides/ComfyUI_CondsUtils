import torch
import os
import json
import folder_paths
import comfy.utils

from comfy.cli_args import args


class SaveReduxEmb:
    def __init__(self):
        # Get the list of base embedding directories
        # self.output_dir will be a LIST of paths
        self.output_dir = folder_paths.get_folder_paths("embeddings")
        # REMOVED: os.makedirs(self.output_dir, exist_ok=True) - This was incorrect

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "filename_prefix": ("STRING", {"default": "redux_embeddings/ComfyUI_style_embed"}),
                             }
               }
    RETURN_TYPES = ()
    FUNCTION = "save_redux_emb"
    OUTPUT_NODE = True

    CATEGORY = "conditioning/style_model"

    def save_redux_emb(self, style_model, clip_vision_output, filename_prefix):
        # --- Prepare Embedding Data ---
        metadata = {}
        cond_flattened = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)
        flattened_shape = (cond_flattened.shape[0], cond_flattened.shape[1])
        cond_emb = torch.zeros(flattened_shape, dtype=torch.float32)
        cond_emb[:cond_flattened.shape[0], :cond_flattened.shape[1]] = cond_flattened.to(dtype=torch.float32)
        redux_emb = {"redux": cond_emb}

        # --- File Naming and Saving Logic ---
        # Split the prefix into directory and base filename components
        base_path, filename_base = os.path.split(filename_prefix)

        # Use the *first* path from the list of embedding directories
        # This is the standard convention in ComfyUI for output nodes
        primary_output_dir = self.output_dir[0]

        # Construct the full folder path relative to the *primary* embeddings directory
        full_output_folder = os.path.join(primary_output_dir, base_path)

        # Ensure the target sub-directory exists (relative to the primary output dir)
        # This call IS necessary here to handle subdirs in filename_prefix
        os.makedirs(full_output_folder, exist_ok=True)

        counter = 1
        max_counter = 99999

        while True:
            save_filename = f"{filename_base}_redux_{counter:05}.safetensors"
            save_path = os.path.join(full_output_folder, save_filename)

            if not os.path.exists(save_path):
                break

            counter += 1

            if counter > max_counter:
                print(f"Warning: Could not find a unique filename for prefix '{filename_prefix}' after {max_counter} attempts.")
                return {"ui": {"text": [f"Error: Max file number reached for {filename_prefix}"]}}

        print(f"Saving Redux embedding to: {save_path}")
        comfy.utils.save_torch_file(redux_emb, save_path, metadata=metadata)

        return {}

class LoadReduxEmb:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "embedding_name": (folder_paths.get_filename_list("embeddings"), ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                             "strength_type": (["multiply", "attn_bias"], ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_reduxembed"

    CATEGORY = "conditioning/style_model"

    def apply_reduxembed(self, conditioning, embedding_name, strength, strength_type):
        embedding_path = folder_paths.get_full_path_or_raise("embeddings", embedding_name)
        cond_embed = comfy.utils.load_torch_file(embedding_path)
        cond = cond_embed["redux"].unsqueeze(dim=0)
        # cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            # even if the strength is 1.0 (i.e, no change), if there's already a mask, we have to add to it
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                # math.log raises an error if the argument is zero
                # torch.log returns -inf, which is what we want
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                # get the size of the mask image
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                # grab the existing mask
                mask = keys.get("attention_mask", None)
                # create a default mask if it doesn't exist
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                # convert the mask dtype, because it might be boolean
                # we want it to be interpreted as a bias
                if mask.dtype == torch.bool:
                    # log(True) = log(1) = 0
                    # log(False) = log(0) = -inf
                    mask = torch.log(mask.to(dtype=torch.float16))
                # now we make the mask bigger to add space for our new tokens
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                # copy over the old mask, in quandrants
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                # now fill in the attention bias to our redux tokens
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)

class SaveCondsEmb:
    def __init__(self):
        # Get the list of base embedding directories
        # self.output_dir will be a LIST of paths
        self.output_dir = folder_paths.get_folder_paths("embeddings")
        # REMOVED: os.makedirs(self.output_dir, exist_ok=True) - This was incorrect

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "filename_prefix": ("STRING", {"default": "t5xxl_embeddings/ComfyUI_T5XXL_embed"}),
                             }
               }
    RETURN_TYPES = ()
    FUNCTION = "save_conds_emb"
    OUTPUT_NODE = True

    CATEGORY = "conditioning/advanced"

    def save_conds_emb(self, conditioning, filename_prefix):
        # --- Prepare Embedding Data ---
        metadata = {}
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            txt = txt.flatten(start_dim=0, end_dim=1)
            txt_shape = (txt.shape[0], txt.shape[1])
            cond_emb = torch.zeros(txt_shape, dtype=torch.float32)
            cond_emb.to(dtype=torch.float32).contiguous()
            cond_emb[:txt.shape[0], :txt.shape[1]] = txt.to(dtype=torch.float32).contiguous()
            t5xxl_emb = {"t5xxl": cond_emb}
            base_path, filename_base = os.path.split(filename_prefix)
            primary_output_dir = self.output_dir[0]
            full_output_folder = os.path.join(primary_output_dir, base_path)
            os.makedirs(full_output_folder, exist_ok=True)

            counter = 1
            max_counter = 99999

            while True:
                save_filename = f"{filename_base}_t5xxl_{counter:05}.safetensors"
                save_path = os.path.join(full_output_folder, save_filename)

                if not os.path.exists(save_path):
                    break

                counter += 1

                if counter > max_counter:
                    print(f"Warning: Could not find a unique filename for prefix '{filename_prefix}' after {max_counter} attempts.")
                    return {"ui": {"text": [f"Error: Max file number reached for {filename_prefix}"]}}

            print(f"Saving T5XXL embedding to: {save_path}")
            comfy.utils.save_torch_file(t5xxl_emb, save_path, metadata=metadata)

        return {}

class LoadT5XXLEmb:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "embedding_name": (folder_paths.get_filename_list("embeddings"), ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                             "strength_type": (["multiply", "attn_bias"], ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_t5xxlembed"

    CATEGORY = "conditioning/advanced"

    def apply_t5xxlembed(self, conditioning, embedding_name, strength, strength_type):
        embedding_path = folder_paths.get_full_path_or_raise("embeddings", embedding_name)
        cond_embed = comfy.utils.load_torch_file(embedding_path)
        cond = cond_embed["t5xxl"].unsqueeze(dim=0)
        # cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            # even if the strength is 1.0 (i.e, no change), if there's already a mask, we have to add to it
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                # math.log raises an error if the argument is zero
                # torch.log returns -inf, which is what we want
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                # get the size of the mask image
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                # grab the existing mask
                mask = keys.get("attention_mask", None)
                # create a default mask if it doesn't exist
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                # convert the mask dtype, because it might be boolean
                # we want it to be interpreted as a bias
                if mask.dtype == torch.bool:
                    # log(True) = log(1) = 0
                    # log(False) = log(0) = -inf
                    mask = torch.log(mask.to(dtype=torch.float16))
                # now we make the mask bigger to add space for our new tokens
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                # copy over the old mask, in quandrants
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                # now fill in the attention bias to our redux tokens
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)


NODE_CLASS_MAPPINGS = {
    "SaveReduxEmb": SaveReduxEmb,
    "LoadReduxEmb": LoadReduxEmb,
    "SaveCondsEmb": SaveCondsEmb,
    "LoadT5XXLEmb": LoadT5XXLEmb,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveReduxEmb": "Save Redux Embedding",
    "LoadReduxEmb": "Load and Apply Redux Embedding",
    "SaveCondsEmb": "Save T5XXL conds as Embedded prompt",
    "LoadT5XXLEmb": "Load and Apply saved T5XXL Embedding",
}
