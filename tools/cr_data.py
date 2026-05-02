import os
import json
from cityscapesscripts.helpers.labels import labels

def convert_cityscapes_to_coco_full(gt_dir, save_path):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 1. Берем ВСЕ категории, кроме совсем мусорных (void, unlabeled)
    # Нам нужны и 'road', 'sky', 'sidewalk', и 'car'
    added_cat_ids = set()
    for label in labels:
        if label.id >= 0 and label.name != "unlabeled": # Игнорируем только технические метки
            coco_data["categories"].append({
                "id": label.id,
                "name": label.name,
                "supercategory": label.category
            })
            added_cat_ids.add(label.name)

    ann_id = 1
    img_id = 1
    
    val_dir = os.path.join(gt_dir, "val")
    if not os.path.exists(val_dir):
        print(f"Ошибка: Путь {val_dir} не найден!")
        return

    for city in os.listdir(val_dir):
        city_path = os.path.join(val_dir, city)
        if not os.path.isdir(city_path): continue
        
        for file in os.listdir(city_path):
            if file.endswith("polygons.json"):
                with open(os.path.join(city_path, file)) as f:
                    data = json.load(f)
                
                img_name = file.replace("_gtFine_polygons.json", "_leftImg8bit.png")
                coco_data["images"].append({
                    "id": img_id,
                    "file_name": os.path.join(city, img_name),
                    "width": data["imgWidth"],
                    "height": data["imgHeight"]
                })
                
                for obj in data["objects"]:
                    label_name = obj["label"]
                    
                    # Проверяем, есть ли такой лейбл в нашем списке категорий
                    cat = next((item for item in coco_data["categories"] if item["name"] == label_name), None)
                    
                    if cat:
                        # Превращаем список точек [[x,y], [x,y]] в плоский список [x, y, x, y]
                        poly = [p for pair in obj["polygon"] for p in pair]
                        
                        # ВАЖНО: COCO требует минимум 3 точки (6 координат)
                        if len(poly) >= 6:
                            coco_data["annotations"].append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": cat["id"],
                                "segmentation": [poly],
                                "iscrowd": 0,
                                "area": 0 # Для расчетов mIoU библиотеками это поле обычно пересчитывается
                            })
                            ann_id += 1
                img_id += 1

    with open(save_path, 'w') as f:
        json.dump(coco_data, f)
    print(f"Готово! Теперь в аннотации {len(coco_data['annotations'])} объектов.")
    print(f"Категории: {[c['name'] for c in coco_data['categories']]}")

convert_cityscapes_to_coco_full("/app/gtFine", "/app/cityscapes_coco_full.json")