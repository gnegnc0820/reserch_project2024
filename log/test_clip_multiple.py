import numpy as np

def is_point_in_polyhedron(point, faces, eps=1e-9):
    """点が多面体の内部にあるかどうかを確認する関数"""
    # 各面に対してチェック
    for face in faces:
        normal, d = face
        # 点と面の距離を計算
        dist = np.dot(normal, point) + d
        # 点が面の外側にある場合は外部とみなす
        if dist > eps:
            return False
    # すべての面に対して内部にある場合はTrue
    return True

def project_point_to_plane(point, normal, d):
    """点を面に投影する関数"""
    # 点と面の距離を計算
    dist = np.dot(normal, point) + d
    # 投影された点を計算
    projected_point = point - dist * normal
    return projected_point

def clip_point_to_polyhedron(point, vertices):
    """点を多面体の範囲内にクリップする関数"""
    # 多面体の面を計算
    faces = []
    n = len(vertices)
    for i in range(n):
        # 次の頂点の組み合わせで三角形の面を作る
        j = (i + 1) % n
        k = (i + 2) % n
        
        # 頂点の組み合わせから法線ベクトルを計算
        a = vertices[i]
        b = vertices[j]
        c = vertices[k]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)  # 正規化
        
        # 面の方程式の定数項を計算
        d = -np.dot(normal, a)
        faces.append((normal, d))
    
    # 点が多面体の内部にあるか確認
    if is_point_in_polyhedron(point, faces):
        return point
    
    # 点が外部にある場合、多面体の面に投影
    min_dist = float('inf')
    clipped_point = None
    
    for normal, d in faces:
        # 点を面に投影
        projected_point = project_point_to_plane(point, normal, d)
        # 投影した点と元の点の距離を計算
        dist = np.linalg.norm(point - projected_point)
        
        # 最も近い投影点を更新
        if dist < min_dist:
            min_dist = dist
            print(dist)
            clipped_point = projected_point
    
    return clipped_point

# 頂点を定義（例：四面体の頂点）
vertices = [
    np.array([1.0, 1.0, 1.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 1.0]),
    np.array([0.0, 1.0, 1.0]),    
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
]

# クリップしたい点の座標を定義
point = np.array([2.0, 2.0, 2.0])

# 点を多面体の範囲内にクリップ
clipped_point = clip_point_to_polyhedron(point, vertices)

print(f"Original point: {point}")
print(f"Clipped point: {clipped_point}")

import matplotlib.pyplot as plt

# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 散布図のプロット
for p in vertices:
    ax.scatter(p[0], p[1], p[2], c='blue', marker='o', s=20)

p = clipped_point
ax.scatter(p[0], p[1], p[2], c='red', marker='o', s=20)

p = point
ax.scatter(p[0], p[1], p[2], c='green', marker='o', s=20)

# 軸のラベルを設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# タイトルを設定
ax.set_title('3D Scatter Plot')

# プロットの表示
plt.show()