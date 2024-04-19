import numpy as np

def calculate_lattice_vertices(lattice_vectors):
    """格子ベクトルから格子の各頂点の座標を計算する関数"""
    # 格子ベクトルの数（次元数）を取得
    n = len(lattice_vectors)
    
    # 各頂点の座標を格納するリスト
    vertices = []
    
    # 2^n個の頂点の座標を計算
    # 0から2^n - 1までの整数を2進数に変換し、各ビットを使用して格子ベクトルを重み付け
    for i in range(2 ** n):
        # 重み付けした格子ベクトルの線形結合
        vertex = np.zeros(n)
        for j in range(n):
            # `i`の`j`番目のビットが1かどうかを確認
            if (i & (1 << j)) != 0:
                # 格子ベクトルを足します
                vertex += lattice_vectors[j]
        vertices.append(vertex)
    
    return vertices

# 例：2次元空間の格子ベクトル
lattice_vectors = [
    np.array([1.0, 0.0]),  # 格子ベクトルa1
    np.array([0.0, 1.0])   # 格子ベクトルa2
]

# 格子の頂点を計算
vertices = calculate_lattice_vertices(lattice_vectors)

# 結果の表示
print("Vertices of the lattice:")
for vertex in vertices:
    print(vertex)
