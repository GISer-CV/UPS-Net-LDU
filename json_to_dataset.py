import os
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed # 改用线程池
from tqdm import tqdm

# ================= 你的路径 =================
JSON_DIR = r"VOCdevkit/VOC2007/j"
MASK_DIR = r"VOCdevkit/VOC2007/jj"
CLASS_MAPPING = {
    "1": 1
}
# ===========================================

def convert_one_file(json_file):
    """
    单个文件的转换逻辑
    """
    try:
        json_path = os.path.join(JSON_DIR, json_file)
        save_name = os.path.splitext(json_file)[0] + ".png"
        save_path = os.path.join(MASK_DIR, save_name)

        # 1. 极速读取 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 2. 获取尺寸 (直接读数据，不计算)
        h = data['imageHeight']
        w = data['imageWidth']
        
        # 3. 准备画布
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 4. 只有当确实有目标时才进行绘制操作，节省 CPU
        shapes = data.get('shapes', [])
        if shapes:
            for shape in shapes:
                label = shape['label']
                if label in CLASS_MAPPING:
                    val = CLASS_MAPPING[label]
                    # 转换坐标 (这一步其实很快)
                    pts = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], color=val)

        # 5. 【关键优化】写入硬盘
        # 使用线程池时，这里是主要瓶颈，必须用 imencode 绕过中文路径bug
        # 压缩率设为 0 (不压缩)，以空间换时间，速度最快
        success, encoded_img = cv2.imencode('.png', mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if success:
            encoded_img.tofile(save_path)
            
        return True

    except Exception as e:
        return f"❌ {json_file}: {e}"

def main():
    if not os.path.exists(MASK_DIR):
        os.makedirs(MASK_DIR)

    # 1. 扫描文件
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    total = len(json_files)
    
    if total == 0:
        print("未找到 JSON 文件！")
        return

    # 2. 设置线程数
    # 线程比进程轻量得多，可以设得比 CPU 核心数多
    # 如果你的电脑是 4 核，设 8 或 16 都可以，因为大部分时间在等硬盘读写
    max_workers = 16 
    
    print(f"🚀 启动轻量级极速模式 (线程数: {max_workers})")
    print(f"📂 待处理文件: {total} 个")

    # 3. 并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(convert_one_file, f) for f in json_files]
        
        # 进度条
        for future in tqdm(as_completed(futures), total=total, unit="img"):
            result = future.result()
            if result is not True:
                print(result)

    print(f"\n✅ 全部搞定！请查看: {MASK_DIR}")

if __name__ == "__main__":
    main()
