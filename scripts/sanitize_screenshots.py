"""
FinRL-X-MT5: Automated Screenshot Sanitizer
Masks private prop firm account credentials, live ticket numbers,
and specific broker server identifiers for public documentation.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def sanitize_screenshots():
    project_root = Path(__file__).resolve().parent.parent
    assets_dir = project_root / "docs" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    font_bold_13 = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 13)
    font_bold_9 = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 9)
    font_mono_11 = ImageFont.truetype("C:\\Windows\\Fonts\\consola.ttf", 11)
    
    for mode in ["Dark", "White"]:
        source_path = project_root / f"{mode}-mode.jpg"
        if not source_path.exists():
            print(f"Warning: {source_path} not found.")
            continue
            
        img = Image.open(source_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        if mode == "Dark":
            # 1. Account Badge Pill (fill inside pill without touching green dot at 380-390 or pill border)
            pill_fill = (21, 28, 46)
            draw.rectangle([(391, 16), (592, 41)], fill=pill_fill)
            draw.text((398, 21), "PropFirm-Live", fill=(255, 255, 255), font=font_bold_13)
            draw.text((495, 21), "#••••••••", fill=(56, 189, 248), font=font_bold_13)
            
            # 2. Guardian Card Preset Tag (x: 365 to 535, y: 598 to 615)
            card_fill = (14, 18, 29)
            draw.rectangle([(365, 598), (535, 615)], fill=card_fill)
            draw.text((368, 602), "PROP FIRM ULTRA-SAFE PRESET", fill=(100, 116, 139), font=font_bold_9)
            
            # 3. Active Positions Table (Smooth Gaussian Blur + Privacy Shield Overlay)
            pos_box = (555, 660, 1075, 755)
            pos_crop = img.crop(pos_box).filter(ImageFilter.GaussianBlur(radius=8))
            img.paste(pos_crop, pos_box)
            
            badge_rect = [(720, 694), (915, 722)]
            draw.rounded_rectangle(badge_rect, radius=6, fill=(15, 23, 42), outline=(51, 65, 85), width=1)
            draw.text((732, 701), "🔒 LIVE POSITIONS MASKED", fill=(148, 163, 184), font=font_bold_9)
            
            # 4. Header Balances (Entire block: labels + numbers)
            hdr_fill = (14, 18, 29)
            draw.rectangle([(1360, 10), (1625, 45)], fill=hdr_fill)
            draw.text((1368, 12), "EQUITY", fill=(148, 163, 184), font=font_bold_9)
            draw.text((1368, 23), "$10,000.00", fill=(255, 255, 255), font=font_bold_13)
            
            draw.text((1460, 12), "BALANCE", fill=(148, 163, 184), font=font_bold_9)
            draw.text((1460, 23), "$10,000.00", fill=(255, 255, 255), font=font_bold_13)
            
            draw.text((1550, 12), "FLOATING PNL", fill=(148, 163, 184), font=font_bold_9)
            draw.text((1560, 23), "+$0.00", fill=(52, 211, 153), font=font_bold_13)
            
            # 5. Total Drawdown Dollar Amount (x: 232 to 325, y: 830 to 848)
            draw.rectangle([(232, 830), (325, 848)], fill=card_fill)
            draw.text((236, 832), "$0.00 / $400", fill=(52, 211, 153), font=font_mono_11)
            
        else:
            # White Mode
            # 1. Account Badge Pill
            pill_fill = (241, 244, 249)
            draw.rectangle([(391, 16), (592, 41)], fill=pill_fill)
            draw.text((398, 21), "PropFirm-Live", fill=(30, 41, 59), font=font_bold_13)
            draw.text((495, 21), "#••••••••", fill=(2, 132, 199), font=font_bold_13)
            
            # 2. Guardian Card Preset Tag
            card_fill = (255, 255, 255)
            draw.rectangle([(365, 598), (535, 615)], fill=card_fill)
            draw.text((368, 602), "PROP FIRM ULTRA-SAFE PRESET", fill=(148, 163, 184), font=font_bold_9)
            
            # 3. Active Positions Table
            pos_box = (555, 660, 1075, 755)
            pos_crop = img.crop(pos_box).filter(ImageFilter.GaussianBlur(radius=8))
            img.paste(pos_crop, pos_box)
            
            badge_rect = [(720, 694), (915, 722)]
            draw.rounded_rectangle(badge_rect, radius=6, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
            draw.text((732, 701), "🔒 LIVE POSITIONS MASKED", fill=(100, 116, 139), font=font_bold_9)
            
            # 4. Header Balances
            hdr_fill = (255, 255, 255)
            draw.rectangle([(1360, 10), (1625, 45)], fill=hdr_fill)
            draw.text((1368, 12), "EQUITY", fill=(100, 116, 139), font=font_bold_9)
            draw.text((1368, 23), "$10,000.00", fill=(15, 23, 42), font=font_bold_13)
            
            draw.text((1460, 12), "BALANCE", fill=(100, 116, 139), font=font_bold_9)
            draw.text((1460, 23), "$10,000.00", fill=(15, 23, 42), font=font_bold_13)
            
            draw.text((1550, 12), "FLOATING PNL", fill=(100, 116, 139), font=font_bold_9)
            draw.text((1560, 23), "+$0.00", fill=(16, 185, 129), font=font_bold_13)
            
            # 5. Total Drawdown Dollar Amount
            draw.rectangle([(232, 830), (325, 848)], fill=card_fill)
            draw.text((236, 832), "$0.00 / $400", fill=(16, 185, 129), font=font_mono_11)
            
        dest_file = assets_dir / f"dashboard_{mode.lower()}_sanitized.png"
        img.save(dest_file, "PNG", optimize=True)
        print(f"[OK] Successfully created: {dest_file}")

if __name__ == "__main__":
    sanitize_screenshots()
