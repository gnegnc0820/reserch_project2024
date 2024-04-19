import numpy as np

def is_point_in_polygon(point, polygon):
    """点が多角形の内部にあるかどうかを確認する関数"""
    n = len(polygon)
    inside = False
    x, y = point
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def project_point_to_line_segment(p, a, b):
    """点を線分上に投影する関数"""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    return a + t * ab

def clip_point_to_polygon(point, polygon):
    """与えられた点を多角形の範囲にクリップする関数"""
    if is_point_in_polygon(point, polygon):
        # 点が多角形の内部にある場合、変更不要
        return point

    # 多角形の各辺に点を投影し、最も近い点を見つける
    min_dist = float('inf')
    clipped_point = None
    for i in range(len(polygon)):
        a = np.array(polygon[i])
        b = np.array(polygon[(i + 1) % len(polygon)])
        proj = project_point_to_line_segment(point, a, b)
        dist = np.linalg.norm(point - proj)
        
        # 最も近い点を更新
        if dist < min_dist:
            min_dist = dist
            clipped_point = proj
            
    return clipped_point

# 多角形の頂点座標を定義（例：正六角形）
polygon = [
    np.array([0.0, 0.0]),
    np.array([5.0, 0.0]),
    np.array([7.5, 4.33]),
    np.array([5.0, 8.66]),
    np.array([0.0, 8.66]),
    np.array([-2.5, 4.33])
]
polygon = [
    np.array([0.0, 0.0]),
    np.array([5.0, 0.0]),
    np.array([7.5, 4.33]),
]


# クリップしたい点の座標を定義
point = np.array([2.0, 4.0])

# 点を多角形の範囲にクリップ
clipped_point = clip_point_to_polygon(point, polygon)

print(f"Original point: {point}")
print(f"Clipped point: {clipped_point}")

import matplotlib.pyplot as plt

plt.scatter(point[0],point[1],c="green")
plt.scatter(clipped_point[0],clipped_point[1],c="red")

for p in polygon:
    plt.scatter(p[0],p[1],c="blue")

plt.show()