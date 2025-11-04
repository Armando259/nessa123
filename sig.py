from PIL import Image
import cv2
import numpy as np
import os

############################################
########### ovaj dio moras mjenjat #########
############################################

mapa = 'slike/'
putanja_novog_loga = 'sigma-certus-logo-black.png'
putanja_starog_loga = 'stari_logo.png' ##ovaj cak ne rabis msc 

############################################
############################################
# Učitaj logove
novi_logo_original = Image.open(putanja_novog_loga)
stari_logo_pil = Image.open(putanja_starog_loga)

# Ako logo ima alpha kanal (prozirnost), konvertiraj u RGB s bijelom pozadinom
if stari_logo_pil.mode == 'RGBA':
    background = Image.new('RGB', stari_logo_pil.size, (255, 255, 255))
    background.paste(stari_logo_pil, mask=stari_logo_pil.split()[3])
    stari_logo_pil = background

# Konvertiraj u OpenCV format
stari_logo_array = np.array(stari_logo_pil)
stari_logo_template = cv2.cvtColor(stari_logo_array, cv2.COLOR_RGB2BGR)
template_gray = cv2.cvtColor(stari_logo_template, cv2.COLOR_BGR2GRAY)
h_template, w_template = template_gray.shape

print(f"Dimenzije novog loga (original): {novi_logo_original.size}")
print(f"Dimenzije starog loga (template): {w_template}x{h_template}")
print("=" * 60)

for ime_slike in os.listdir(mapa):
    if ime_slike.endswith('.jpg') or ime_slike.endswith('.png'):
        putanja = os.path.join(mapa, ime_slike)
        print(f"\n🔍 Obrađujem: {ime_slike}")
        
        slika_cv = cv2.imread(putanja)
        if slika_cv is None:
            print(f"❌ Greška pri učitavanju: {ime_slike}")
            continue
            
        h_img, w_img = slika_cv.shape[:2]
        print(f"   Dimenzije slike: {w_img}x{h_img}")
        
        slika_gray = cv2.cvtColor(slika_cv, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale template matching
        best_match_val = 0
        best_match_loc = None
        best_match_size = None
        best_scale = 1.0
        
        scales = np.linspace(0.5, 1.5, 40)
        print(f"   Tražim logo u {len(scales)} različitih veličina...")
        
        for scale in scales:
            new_w = int(w_template * scale)
            new_h = int(h_template * scale)
            
            if new_w > w_img or new_h > h_img:
                continue
            
            resized_template = cv2.resize(template_gray, (new_w, new_h))
            result = cv2.matchTemplate(slika_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            y_pos = max_loc[1]
            x_pos = max_loc[0]
            
            # Filter: donji lijevi kut
            if y_pos > h_img * 0.5 and x_pos < w_img * 0.3:
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_loc = max_loc
                    best_match_size = (new_w, new_h)
                    best_scale = scale
        
        print(f"   Najbolja sličnost: {best_match_val:.4f}")
        print(f"   Najbolja skala: {best_scale:.2f}x ({best_match_size})" if best_match_size else "   Nije pronađena validna skala")
        print(f"   Pozicija: {best_match_loc}")
        
        threshold = 0.6
        
        if best_match_val > threshold and best_match_loc is not None:
            w_match, h_match = best_match_size
            top_left = best_match_loc
            bottom_right = (top_left[0] + w_match, top_left[1] + h_match)
            
            print(f"   ✅ STARI LOGO DETEKTIRAN!")
            print(f"   📏 Koordinate: {top_left} -> {bottom_right}")
            print(f"   📐 Dimenzije starog loga: {w_match}x{h_match}")
            
            # Obriši stari logo
            cv2.rectangle(slika_cv, top_left, bottom_right, (255, 255, 255), -1)
            print(f"   🗑️  Stari logo OBRISAN")
            
            # Konvertiraj u PIL
            slika_cv_rgb = cv2.cvtColor(slika_cv, cv2.COLOR_BGR2RGB)
            slika = Image.fromarray(slika_cv_rgb)
            
            # KLJUČNO: Skaliraj novi logo na IDENTIČNE dimenzije kao stari
            novi_logo_scaled = novi_logo_original.resize((w_match, h_match), Image.LANCZOS)
            print(f"   📐 Novi logo skaliran na: {w_match}x{h_match}")
            
            # Postavi na ISTU poziciju kao stari logo
            pozicija = top_left
            print(f"   📍 Dodajem novi logo na poziciju: {pozicija}")
            
            # Ako novi logo ima prozirnost, koristi je
            if novi_logo_scaled.mode == 'RGBA':
                slika.paste(novi_logo_scaled, pozicija, novi_logo_scaled)
            else:
                slika.paste(novi_logo_scaled, pozicija)
            
            print(f"   ✅ Novi logo postavljen na istu poziciju i dimenziju!")
            
        else:
            print(f"   ⚠️  Logo NIJE detektiran")
            print(f"      Razlog: sličnost {best_match_val:.4f} < threshold {threshold}")
            if best_match_loc is None:
                print(f"      Ili nije pronađen u donjem lijevom kutu")
            slika = Image.open(putanja)
        
        # Spremi kao WebP
        novo_ime = os.path.splitext(putanja)[0] + '.webp'
        slika.save(novo_ime, format='webp', quality=85)
        print(f"   💾 Spremljeno kao: {novo_ime}")

print("\n" + "=" * 60)
print("✨ Obrada završena!")
print("\n" + "=" * 60)
print("\n")
print("    ╔════════════════════════════════════════╗")
print("    ║                                        ║")
print("    ║         ❤️  JA TEBE VOLIM  ❤️         ║")
print("    ║                                        ║")
print("    ╚════════════════════════════════════════╝")
print("\n")
print("=" * 60)
print("✨ Obrada završena! ✨")


