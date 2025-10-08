import torch
import os
import json
import folder_paths
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator
from comfy.cli_args import args

folder_paths.add_model_folder_path("conds", os.path.join(folder_paths.models_dir, "conds"), is_default=True)

class SaveReduxEmb:
    def __init__(self):
        self.output_dir = folder_paths.get_folder_paths("conds")

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
        metadata = {}
        cond_flattened = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)
        flattened_shape = (cond_flattened.shape[0], cond_flattened.shape[1])
        cond_emb = torch.zeros(flattened_shape, dtype=torch.float32)
        cond_emb[:cond_flattened.shape[0], :cond_flattened.shape[1]] = cond_flattened.to(dtype=torch.float32)
        redux_emb = {"redux": cond_emb}
        base_path, filename_base = os.path.split(filename_prefix)
        primary_output_dir = self.output_dir[0]
        full_output_folder = os.path.join(primary_output_dir, base_path)
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
                             "embedding_name": (folder_paths.get_filename_list("conds"), ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                             "strength_type": (["multiply", "attn_bias"], ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_reduxembed"

    CATEGORY = "conditioning/style_model"

    def apply_reduxembed(self, conditioning, embedding_name, strength, strength_type):
        embedding_path = folder_paths.get_full_path_or_raise("conds", embedding_name)
        cond_embed = comfy.utils.load_torch_file(embedding_path)
        cond = cond_embed["redux"].unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                mask = keys.get("attention_mask", None)
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                if mask.dtype == torch.bool:
                    mask = torch.log(mask.to(dtype=torch.float16))
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)

class SaveCondsEmb:
    def __init__(self):
        self.output_dir = folder_paths.get_folder_paths("conds")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "filename_prefix": ("STRING", {"default": "t5xxl_conds/ComfyUI_T5XXL_conds"}),
                             }
               }
    RETURN_TYPES = ()
    FUNCTION = "save_conds"
    OUTPUT_NODE = True

    CATEGORY = "conditioning/advanced"

    def save_conds(self, conditioning, filename_prefix):
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

            print(f"Saving T5XXL conds to: {save_path}")
            comfy.utils.save_torch_file(t5xxl_emb, save_path, metadata=metadata)

        return {}

class LoadT5XXLEmb:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "embedding_name": (folder_paths.get_filename_list("conds"), ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                             "strength_type": (["multiply", "attn_bias"], ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_t5xxl_conds"

    CATEGORY = "conditioning/advanced"

    def apply_t5xxl_conds(self, conditioning, embedding_name, strength, strength_type):
        embedding_path = folder_paths.get_full_path_or_raise("conds", embedding_name)
        cond_embed = comfy.utils.load_torch_file(embedding_path)
        cond = cond_embed["t5xxl"].unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                mask = keys.get("attention_mask", None)
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                if mask.dtype == torch.bool:
                    mask = torch.log(mask.to(dtype=torch.float16))
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)

class LoadT5XXLConds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"embedding_name": (folder_paths.get_filename_list("conds"), ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_t5xxl_conds"

    CATEGORY = "conditioning/advanced"

    def load_t5xxl_conds(self, embedding_name):
        embedding_path = folder_paths.get_full_path_or_raise("conds", embedding_name)
        cond_embed = comfy.utils.load_torch_file(embedding_path)
        c_out = []

        cond = cond_embed["t5xxl"].unsqueeze(dim=0)

        c_out.append([cond, {"pooled_output": None}])

        return (c_out,)

class InsertT5XXLEmb:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "embedding_name": (folder_paths.get_filename_list("conds"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "strength_type": (["attn_bias", "multiply"],),
                "insert_at_index": ("INT", {"default": 1, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "insert_t5xxl_conds"

    CATEGORY = "conditioning/advanced"

    def insert_t5xxl_conds(self, conditioning, embedding_name, strength, strength_type, insert_at_index):
        embedding_path = folder_paths.get_full_path_or_raise("conds", embedding_name)
        cond_embed = comfy.utils.load_torch_file(embedding_path)
        raw_cond = cond_embed["t5xxl"].unsqueeze(dim=0)
        cond = raw_cond[:, :-1, :]
        if strength_type == "multiply":
            cond *= strength
        n_cond = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            cond = cond.to(txt.device, dtype=txt.dtype)
            n_txt_original = txt.shape[1]
            safe_insert_at_index = min(insert_at_index, n_txt_original)
            txt_part1 = txt[:, :safe_insert_at_index, :]
            txt_part2 = txt[:, safe_insert_at_index:, :]
            new_txt = torch.cat((txt_part1, cond, txt_part2), dim=1)
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                attn_bias = torch.log(torch.tensor(strength if strength_type == "attn_bias" else 1.0, dtype=torch.float16))
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                mask = keys.get("attention_mask", None)
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt_original + n_ref, n_txt_original + n_ref), dtype=torch.float16)
                if mask.dtype == torch.bool:
                    mask = torch.log(mask.to(dtype=torch.float16))
                n_part1 = txt_part1.shape[1]
                n_part2 = txt_part2.shape[1]
                n_new_txt = new_txt.shape[1]
                new_mask_size = n_new_txt + n_ref
                new_mask = torch.zeros((txt.shape[0], new_mask_size, new_mask_size), dtype=torch.float16)
                p1_end = n_part1
                cond_end = n_part1 + n_cond
                p2_end = n_new_txt
                new_mask[:, :p1_end, :p1_end] = mask[:, :n_part1, :n_part1]
                new_mask[:, :p1_end, cond_end:p2_end] = mask[:, :n_part1, n_part1:n_txt_original]
                new_mask[:, cond_end:p2_end, :p1_end] = mask[:, n_part1:n_txt_original, :n_part1]
                new_mask[:, cond_end:p2_end, cond_end:p2_end] = mask[:, n_part1:n_txt_original, n_part1:n_txt_original]
                if n_ref > 0:
                    new_mask[:, p2_end:, :p1_end] = mask[:, n_txt_original:, :n_part1]
                    new_mask[:, p2_end:, cond_end:p2_end] = mask[:, n_txt_original:, n_part1:n_txt_original]
                    new_mask[:, :p1_end, p2_end:] = mask[:, :n_part1, n_txt_original:]
                    new_mask[:, cond_end:p2_end, p2_end:] = mask[:, n_part1:n_txt_original, n_txt_original:]
                    new_mask[:, p2_end:, p2_end:] = mask[:, n_txt_original:, n_txt_original:]
                new_mask[:, :, p1_end:cond_end] = attn_bias
                new_mask[:, p1_end:cond_end, :] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size
            c_out.append([new_txt, keys])
        return (c_out,)

class TextEncodeEmbedding(ComfyNodeABC):
    def __init__(self):
        self.output_dir = folder_paths.get_folder_paths("conds")

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_embedding"

    OUTPUT_NODE = True

    CATEGORY = "_for_testing"

    def save_embedding(self, samples, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

NODE_CLASS_MAPPINGS = {
    "SaveReduxEmb": SaveReduxEmb,
    "LoadReduxEmb": LoadReduxEmb,
    "SaveCondsEmb": SaveCondsEmb,
    "LoadT5XXLEmb": LoadT5XXLEmb,
    "LoadT5XXLConds": LoadT5XXLConds,
    "InsertT5XXLEmb": InsertT5XXLEmb,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveReduxEmb": "Save Redux Embedding",
    "LoadReduxEmb": "Load and Apply Redux Embedding",
    "SaveCondsEmb": "Save T5XXL conds Pre-computed as Safetensors file",
    "LoadT5XXLEmb": "Load and Apply Pre-computed T5XXL Conds",
    "LoadT5XXLConds": "Load Pre-computed T5XXL Conds",
    "InsertT5XXLEmb": "Load T5XXL Embedding and insert to conds by splicing",
}
