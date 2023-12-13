import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import plotly.express as px 
# from matplotlib.font_manager import FontProperties

# matplotlibのフォント設定（Windowsの場合）
# font_path = "C:/Windows/Fonts/meiryo.ttc"  # 例: Windowsの場合
# # フォントプロパティの設定
# font_prop = FontProperties(fname=font_path, weight='bold') 
# jp_font = FontProperties(fname=font_path)
# plt.rcParams["font.family"] = jp_font.get_name()

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
    df_con = pd.concat([df_train, df_test]).reset_index(drop=True)

    # 浮動小数点型の値を持つカラムのみに絞る
    float_columns = df_con.select_dtypes(include='float').columns
    df = df_con[float_columns]
    

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
    df_time = pd.concat([df_con["DATA_DATETIME"],df],axis=1)
    df_time.loc[:, "DATA_DATETIME"] = pd.to_datetime(df_time["DATA_DATETIME"], format='%Y/%m/%d %H:%M:%S')
    df_time.sort_values(by="DATA_DATETIME", inplace=True)

    # Streamlitアプリケーションの作成
    st.title('マハラノビス距離の計算と主成分分析 (PCA)')

    # データの表示
    st.write('教師データと評価データの結合後のデータフレーム:')
    st.write(df)

    # マハラノビス距離のプロットを散布図に変更
    st.write('マハラノビス距離のプロット:')
    st.scatter_chart(df_time.set_index('DATA_DATETIME')[['mahalanobis_distance']])


    # 主成分数ごとの累積寄与率の折れ線グラフ
    st.write('Cumulative Explained Variance Ratio:')
    fig_variance_ratio = plt.figure()
    plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
    plt.title('Principal Component Analysis (PCA) - Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.xticks(np.arange(1, 8, 1))
    plt.grid(True)
    st.pyplot(fig_variance_ratio)

    # マハラノビス距離のプロット
    st.write('Mahalanobis Distance Plot:')
    fig, ax = plt.subplots()
    ax.scatter(df.index, df['mahalanobis_distance'])
    ax.set_xlabel('Index')
    ax.set_ylabel('Mahalanobis Distance')
    ax.set_title('Distribution of Mahalanobis Distance')

    # 散布図を描画
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.box(False)

    # 散布図を描画
    scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['mahalanobis_distance'], cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.xticks(rotation=25)
    plt.ylabel('Second Principal Component')
    plt.title('Scatter Plot in Principal Component Space')
    plt.colorbar(scatter, label='Mahalanobis Distance')

    # x軸とy軸の交点に"O"を表記（寄せる）
    plt.text(0, 0, 'O',  fontsize=12, ha='left', va='top')

    plt.show()
    # Streamlitにプロットを表示
    st.pyplot(fig)

    # 主成分空間での3D散布図
    st.write('3D Scatter Plot in Principal Component Space:')
    fig_3d = px.scatter_3d(
        df_time.set_index('DATA_DATETIME'),
        x=df_pca[:, 0],
        y=df_pca[:, 1],
        z=df_pca[:, 2],  # 第三主成分をZ軸に追加
        color=df['mahalanobis_distance'],  # マハラノビス距離を色で表現
        size_max=50,
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component', 'z': 'Third Principal Component'},
        title='3D Scatter Plot in Principal Component Space',
        color_continuous_scale='viridis',
    )
    fig_3d.update_layout(scene=dict(xaxis_title='First Principal Component', yaxis_title='Second Principal Component', zaxis_title='Third Principal Component'))
    st.plotly_chart(fig_3d)



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
    # 第三主成分軸に対するLoadingsの絶対値を取得
    loadings_third_pc = np.abs(loadings[:, 2])

    # Loadingsを絶対値でソートしてインデックスを取得
    sorted_loadings_third_pc_indices = np.argsort(loadings_third_pc)[::-1]

    # ソートされたLoadings
    sorted_loadings_third_pc = pd.DataFrame(loadings[:, 2][sorted_loadings_third_pc_indices], index=df.columns[sorted_loadings_third_pc_indices], columns=['Third PC Loadings'])
    filtered_loadings_third_pc = sorted_loadings_third_pc[sorted_loadings_third_pc['Third PC Loadings'].abs() > 0.0001]
    filtered_loadings_third_pc.columns = ['第三主成分負荷量']

    # 第三主成分軸に対する負荷量の表示
    st.write('第三主成分軸に対する負荷量:')
    st.table(filtered_loadings_third_pc)

else:
    st.sidebar.warning("ファイルがアップロードされていません。教師データと評価データを選択してください。")
