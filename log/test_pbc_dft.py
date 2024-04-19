from pyscf import pbc, dft    # type: ignore
# セルオブジェクトの作成
cell = pbc.gto.Cell()

# 原子座標を設定
cell.atom = '''
C 0.0 0.0 0.0
C 1.0 1.0 1.0
'''

print(cell.atom)
exit()

# 格子ベクトルを設定
cell.a = [
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0]
]

# 基底セットと疑似ポテンシャルを設定
cell.basis = 'sto3g'
cell.pseudo = {'C': 'gthbp'}

# k-ポイントを設定
cell.make_kpts([4, 4, 4])

# 計算精度の設定
# cell.omega_cutoff = 50  # カットオフエネルギー
cell.ke_cutoff = 50  # カットオフエネルギー
cell.precision = 1e-6  # 精度

# セルオブジェクトを構築
cell.build()

# DFT計算の設定
mf = dft.RKS(cell)
mf.xc = 'PBE'  # 交換・相関汎関数

# 計算の実行
print("DFT計算の開始")
energy = mf.kernel()
print(f"終了: エネルギー = {energy:.6f} ハートリー")
