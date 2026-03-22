import os
import json
import glob
from safetensors.torch import load_file, save_file

def fix_checkpoint(ckpt_dir):
    print(f"Javítás megkezdése: {ckpt_dir}")
    
    # 1. config.json javítása
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Hozzáadjuk a supported_languages listát (kiolvasva a codec_language_id-ből)
        codec_dict = config.get("talker_config", {}).get("codec_language_id", {})
        supported = ["auto"] + [k for k in codec_dict.keys() if "dialect" not in k]
        config["supported_languages"] = supported
        
        # Visszaállítjuk a helyes architektúra nevet
        config["architectures"] = ["Qwen3TTSModel"]
        config["tts_model_type"] = "base"
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(" - config.json sikeresen javítva!")
    else:
        print(" - HIBA: config.json nem található!")

    # 2. safetensors súlyok javítása (a 'model.' prefix hozzáadása)
    safetensors_files = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
    
    if not safetensors_files:
        print(" - HIBA: Nem található .safetensors fájl!")
        return

    for st_file in safetensors_files:
        print(f" - Súlyok javítása: {os.path.basename(st_file)} ...")
        
        # Betöltjük az eredeti tenzorokat
        tensors = load_file(st_file)
        new_tensors = {}
        
        for key, tensor in tensors.items():
            # Ha hiányzik a 'model.' prefix, hozzáadjuk
            if not key.startswith("model."):
                new_tensors[f"model.{key}"] = tensor
            else:
                new_tensors[key] = tensor
                
        # Eredeti fájl átnevezése biztonsági mentésként (opcionális)
        backup_file = st_file + ".bak"
        os.rename(st_file, backup_file)
        
        # Új, javított tenzorok mentése
        save_file(new_tensors, st_file)
        print(f" - {os.path.basename(st_file)} sikeresen javítva!")
        
        # Ha minden jól ment, letörölhetjük a backupot, hogy ne foglalja a helyet
        os.remove(backup_file)

    print("KÉSZ! A checkpoint most már tökéletesen betölthető.\n")

if __name__ == "__main__":
    # Itt add meg a javítandó mappa elérési útját!
    # Ha több epoch is lement, mindegyikre futtasd le.
    fix_checkpoint("output/checkpoint-epoch-0")
    
    # Ha kész a többi is, azokat is megadhatod:
    # fix_checkpoint("output/checkpoint-epoch-1")
    # fix_checkpoint("output/checkpoint-epoch-2")
