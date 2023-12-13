import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CsvProcessorApp:
    def __init__(self):
        self.df_pqc = None

    def browse_file_pqc(self):
        file_path_pqc = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
        if file_path_pqc:
            self.file_path_var_pqc = file_path_pqc
            st.success("ファイルが正常にアップロードされました.")

    def load_csv_pqc(self):
        if hasattr(self, "file_path_var_pqc"):
            try:
                # Shift-jisでCSVファイルを読み込む
                self.df_pqc = pd.read_csv(self.file_path_var_pqc, encoding="Shift-jis")
                st.success("CSVファイルが正常に読み込まれました.")
            except Exception as e:
                st.error(f"PQC CSV ロードエラー: {str(e)}")

    def display_loaded_data(self):
        if self.df_pqc is not None:
            st.subheader("読み込まれたデータ:")
            st.dataframe(self.df_pqc)

    @st.cache
    def compute_pie_chart_data(self, selected_obj, selected_eqp):
        # 装置IDごとに選択したカラムの合計を計算
        grouped_sum = self.df_pqc.groupby(selected_eqp)[selected_obj].sum()

        # 装置IDごとのレコード数を計算
        grouped_count = self.df_pqc[selected_eqp].value_counts()

        # 新しいDataFrameを作成
        df_grouped = pd.DataFrame({
            selected_obj: grouped_sum,
            'Count': grouped_count
        })

        # 降順にソート
        df_grouped_sorted = df_grouped.sort_values(by=selected_obj, ascending=True)

        return df_grouped_sorted

    def draw_pie_chart(self, selected_obj, selected_eqp):
        # 円グラフのデータを計算
        df_grouped_sorted = self.compute_pie_chart_data(selected_obj, selected_eqp)

        # 円グラフを描画
        fig, ax = plt.subplots(figsize=(10, 10))
        df_grouped_sorted[selected_obj].plot(kind='pie', ax=ax, autopct=lambda p: '{:.1f}%\n(N:{:.0f})'.format(p, (p/100)*df_grouped_sorted['Count'].sum()), startangle=90)
        ax.set_title("装置ID参照")
        ax.set_ylabel('')

        # 各装置IDのレコード数を表示
        st.write(df_grouped_sorted.sort_values(by=selected_obj, ascending=False)['Count'])

# Streamlit Appの開始
def main():
    st.title("Python解析ツール")

    # セッションの状態管理
    if 'app' not in st.session_state:
        st.session_state.app = CsvProcessorApp()

    app = st.session_state.app

    app.browse_file_pqc()
    if st.button("読み込み"):
        app.load_csv_pqc()

    app.display_loaded_data()

    if app.df_pqc is not None:
        selected_obj = st.selectbox("目的変数を選択してください", app.df_pqc.columns)
        selected_eqp = st.selectbox("装置カラムを選択してください", app.df_pqc.columns)
        if st.button("円グラフを描画"):
            app.draw_pie_chart(selected_obj, selected_eqp)

if __name__ == "__main__":
    main()
