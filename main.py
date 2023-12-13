import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# matplotlibのフォント設定（Windowsの場合）
font_path = "C:/Windows/Fonts/meiryo.ttc"  # 例: Windowsの場合
# フォントプロパティの設定
font_prop = FontProperties(fname=font_path, weight='bold') 
jp_font = FontProperties(fname=font_path)
plt.rcParams["font.family"] = jp_font.get_name()

# ファイルアップロード
st.sidebar.title("ファイルのアップロード")
uploaded_train_file = st.sidebar.file_uploader("教師データを選択してください", type=["xlsx", "csv"])
uploaded_test_file = st.sidebar.file_uploader("評価データを選択してください", type=["xlsx", "csv"])

# ファイルがアップロードされたか確認
if uploaded_train_file is not None and uploaded_test_file is not None:
    # 教師データ読み込み
    df_train = pd.read_excel(uploaded_train_file)
    # 評価データ読み込み
    df_test = pd.read_excel(uploaded_test_file)
    # 教師データと評価データの結合
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    # 浮動小数点型の値を持つカラムのみに絞る
    float_columns = df.select_dtypes(include='float').columns
    df = df[float_columns]

    # 主成分分析（PCA）を実行
    pca = PCA(n_components=7)  # 主成分数を7までに制限
    df_pca = pca.fit_transform(df)

    # マハラノビス距離の計算
    mean_vector = df_pca.mean(axis=0)
    covariance_matrix = np.cov(df_pca, rowvar=False)
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)

    # Mahalanobis distanceを計算して新しい列に追加
    df['mahalanobis_distance'] = [distance.mahalanobis(x, mean_vector, covariance_matrix_inv) for x in df_pca]

    # 累積寄与率の計算
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    # Loadingsの計算
    loadings = pca.components_.T

    # 第一主成分軸に対するLoadingsの絶対値を取得
    loadings_first_pc = np.abs(loadings[:, 0])

    # Loadingsを絶対値でソートしてインデックスを取得
    sorted_loadings_indices = np.argsort(loadings_first_pc)[::-1]

    # ソートされたLoadings
    sorted_loadings = pd.DataFrame(loadings[:, 0][sorted_loadings_indices], index=df.columns[sorted_loadings_indices], columns=['First PC Loadings'])

    # 第二主成分軸に対するLoadingsの絶対値を取得
    loadings_second_pc = np.abs(loadings[:, 1])

    # Loadingsを絶対値でソートしてインデックスを取得
    sorted_loadings_second_pc_indices = np.argsort(loadings_second_pc)[::-1]

    # ソートされたLoadings
    sorted_loadings_second_pc = pd.DataFrame(loadings[:, 1][sorted_loadings_second_pc_indices], index=df.columns[sorted_loadings_second_pc_indices], columns=['Second PC Loadings'])

    # Streamlitアプリケーションの作成
    st.title('マハラノビス距離の計算と主成分分析 (PCA)')

    # データの表示
    st.write('教師データと評価データの結合後のデータフレーム:')
    st.write(df)

    # マハラノビス距離のプロットを散布図に変更
    st.write('マハラノビス距離のプロット:')
    st.scatter_chart(df[['mahalanobis_distance']])

    # 累積寄与率の折れ線グラフ
    st.write('主成分数ごとの累積寄与率:')
    fig_variance_ratio = plt.figure()
    plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
    plt.title('主成分分析 (PCA) - 累積寄与率')
    plt.xlabel('主成分の数')
    plt.ylabel('累積寄与率')
    plt.xticks(np.arange(1, 8, 1))  # X軸の目盛りを1から7までの自然数に設定
    plt.grid(True)
    st.pyplot(fig_variance_ratio)

        # マハラノビス距離のプロット
    st.write('マハラノビス距離のプロット:')
    fig, ax = plt.subplots()
    ax.scatter(df.index, df['mahalanobis_distance'])
    ax.set_xlabel('インデックス')
    ax.set_ylabel('マハラノビス距離')
    ax.set_title('マハラノビス距離の分布', fontproperties=font_prop)

    # 散布図を描画
    # グラフの軸線を非表示にする
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # グラフの枠を非表示にする
    plt.box(False)

    # 散布図を描画
    scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['mahalanobis_distance'], cmap='viridis')
    plt.xlabel('第一主成分軸')
    plt.xticks(rotation=25)
    plt.ylabel('第二主成分軸')
    plt.title('主成分空間上の散布図')
    plt.colorbar(scatter, label='マハラノビス距離')

    # x軸とy軸の交点に"O"を表記（寄せる）
    plt.text(0, 0, 'O',  fontsize=12, ha='left', va='top') 

    plt.show()


    # Streamlitにプロットを表示
    st.pyplot(fig)

    # 絶対値が0より大きなLoadingsのみを表示
    filtered_loadings = sorted_loadings[sorted_loadings['First PC Loadings'].abs() > 0.0001]
    filtered_loadings_second_pc = sorted_loadings_second_pc[sorted_loadings_second_pc['Second PC Loadings'].abs() > 0.0001]

    # カラム名を日本語に変更
    filtered_loadings.columns = ['第一主成分負荷量']
    filtered_loadings_second_pc.columns = ['第二主成分負荷量']

    # Loadingsの表示
    st.write('第一主成分軸に対する負荷量:')
    st.table(filtered_loadings)

    st.write('第二主成分軸に対する負荷量:')
    st.table(filtered_loadings_second_pc)

else:
    st.sidebar.warning("ファイルがアップロードされていません。教師データと評価データを選択してください。")
