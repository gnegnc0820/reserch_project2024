from pyscf.pbc import gto, dft  # type: ignore

# 炭素原子のDFT計算のためのセルを定義
cell = gto.M(
    atom='C 0 0 0',  # 炭素原子を原子座標に設定
    basis='sto3g',  # 基底関数の設定
    pseudo={'C': 'gthbp'},  # 疑似ポテンシャルの設定
)

# DFT計算の設定（RKS: Restricted Kohn-Sham法）
mf = dft.RKS(cell)
mf.xc = 'PBE'  # 交換・相関汎関数の設定

# SCF収束基準やその他のオプションの設定
mf.conv_tol = 1e-8  # SCF収束の許容誤差

# DFT計算の実行
print("開始: DFT計算")
energy = mf.kernel()
print(f"終了: エネルギー = {energy:.6f} ハートリー")