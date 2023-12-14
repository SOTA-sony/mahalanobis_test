import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import plotly.express as px 
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dash_table
import plotly.graph_objs as go

# アプリケーションの背景色を設定
st.set_page_config(
    page_title="Your App Title",
    page_icon="🌐",
    initial_sidebar_state="expanded",  # サイドバーを開いた状態で開始
)

# サイドバーの背景色をモノクロに設定
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #808080;  /* ホワイト（白）*/
        color: #000000;  /* ブラック（黒）*/
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ここにアプリケーションのコードを追加
# 例：st.title("Hello, Streamlit!")

# サイドバーへのコンテンツの追加
st.sidebar.title("Sidebar Title")

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

    # 主成分分析（PCA）を実行
    pca = PCA(n_components=7)  # 主成分数を7までに制限
    df_pca = pca.fit_transform(df.loc[:,float_columns])

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
    # 日時列と float 列を分ける
   # "DATA_DATETIME" を含む列名を除外して float_columns を作成
    float_columns = df.select_dtypes(include='float').columns.tolist()
    # 日時列と float 列を分ける
    df_time = df[["DATA_DATETIME"] + float_columns].copy()

    # 日時列を datetime フォーマットに変換
    df_time["DATA_DATETIME"] = pd.to_datetime(df_time["DATA_DATETIME"], format='%Y/%m/%d %H:%M:%S')

    # 日時列を基準に DataFrame をソート
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
    st.write('主成分数ごとの累積寄与率の折れ線グラフ：')
    fig_variance_ratio = plt.figure()
    plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
    plt.title('Principal Component Analysis (PCA) - Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.xticks(np.arange(1, 8, 1))
    plt.grid(True)
    st.pyplot(fig_variance_ratio)
    
    # Streamlitアプリケーション
    st.title('2D散布図')

    df_pca_2d = pd.DataFrame(df_pca, columns=["1","2","3","4","5","6","7"])
    df_2d = pd.concat([df[["EES_WAFER_ID","mahalanobis_distance"]],df_pca_2d],axis=1)

    # 散布図
    scatter_data = px.scatter(
        df_2d, x="1", y="2",
        color="mahalanobis_distance",  # マハラノビス距離をカラーに設定
        color_continuous_scale='viridis',
        labels={'1': '第1主成分軸', '2': '第2主成分軸', 'EES_WAFER_ID': 'ウェーハID', 'mahalanobis_distance': 'マハラノビス距離'},
        title='主成分空間上の散布図',
        hover_data={'EES_WAFER_ID': True}  # カーソルを当てた際に表示するデータを指定
    )

    # 垂直および水平の線を追加
    scatter_data.update_layout(
        shapes=[
            dict(type='line', x0=0, x1=0, y0=-1, y1=1, line=dict(color='black', width=2)),  # 垂直線
            dict(type='line', x0=-1, x1=1, y0=0, y1=0, line=dict(color='black', width=2))   # 水平線
        ]
    )

    # 散布図を表示
    st.plotly_chart(scatter_data)

    # クリックされた点の情報
    selected_points_info = []

    # チェックボックスを使用して複数選択
    selected_wafer_ids = st.multiselect("ウェーハIDを選択してテーブルを表示", df_2d["EES_WAFER_ID"])

    # 選択された点の情報を取得
    for wafer_id in selected_wafer_ids:
        selected_points_info.append(df[df["EES_WAFER_ID"] == wafer_id])

    # 選択された点の情報をテーブルで表示
    if selected_points_info:
        selected_point_df = pd.concat(selected_points_info)
        st.table(selected_point_df)

#############################################################################################
    # 主成分空間での3D散布図
    st.title('3D散布図')
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

    # 3D Scatter Plotを表示
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
