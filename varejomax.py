import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
 
 st.set_page_config(page_title="Varejo & Distribuição – KPI & Forecast", layout="wide")
 
 # -----------------------------
 # 1) Dados FICTÍCIOS
 # -----------------------------
 np.random.seed(42)
 dates = pd.date_range(start="2024-10-01", end="2025-09-30", freq="D")
 lojas = ["Loja Centro", "Loja Norte", "Loja Sul"]
 canais = ["Loja Física", "E-commerce"]
 cats = ["Mercearia", "Bebidas", "Frios & Laticínios", "Higiene", "Limpeza"]
 skus = [f"SKU-{i:03d}" for i in range(1, 81)]
 
 # Catálogo SKU->categoria
 sku_cat = np.random.choice(cats, size=len(skus))
 catalogo = pd.DataFrame({"sku": skus, "categoria": sku_cat})
 
 # Gera base diária por loja/canal/SKU
 rows = []
 for d in dates:
     saz = 1.0 + 0.15*np.sin(2*np.pi*(d.timetuple().tm_yday)/365)  # sazonalidade anual suave
     for loja in lojas:
         for canal in canais:
             # Amostra aleatória de 25 SKUs/dia por loja/canal
             sample_skus = np.random.choice(skus, size=25, replace=False)
@@ -134,85 +134,105 @@ st.divider()
 # -----------------------------
 # 5) Tabelas operacionais
 # -----------------------------
 colA, colB = st.columns(2)
 with colA:
     st.subheader("Top 20 SKUs por Margem")
     st.dataframe(sku_view.sort_values("margem", ascending=False).head(20))
 
 with colB:
     st.subheader("Bottom 20 SKUs por Margem")
     st.dataframe(sku_view.sort_values("margem", ascending=True).head(20))
 
 # Rupturas recorrentes (proxy: baixa qtd média e alta variação → sinalizar)
 rupt = (dff.groupby(["loja","sku"])
           .agg(qtd_med=("qtd","mean"), qtd_cv=("qtd", lambda x: x.std()/x.mean() if x.mean() else np.nan))
           .reset_index())
 rupt["alerta_ruptura"] = (rupt["qtd_med"] < 2) | (rupt["qtd_cv"] > 1.2)
 st.subheader("Possíveis Rupturas (proxy estatística)")
 st.dataframe(rupt[rupt["alerta_ruptura"]].dropna().head(50))
 
 st.divider()
 
 # -----------------------------
 # 6) Campo de PREVISIBILIDADE (Forecast semanal por Categoria)
 # -----------------------------
 st.header("Previsibilidade (Forecast) – Regressão Sazonal")
 cat_escolhida = st.selectbox("Categoria para prever", options=sorted(dff["categoria"].unique()))
 horizonte = st.slider("Horizonte (semanas à frente)", 4, 16, 8)
 
 serie_cat = (dff.loc[dff["categoria"]==cat_escolhida]
                 .groupby("data")["vendas"].sum()
                 .resample("W").sum())
 
 if len(serie_cat.dropna()) > 20:
     train = serie_cat.copy().to_frame(name="vendas")
     train["idx"] = np.arange(len(train))
     train["sin52"] = np.sin(2 * np.pi * train["idx"] / 52)
     train["cos52"] = np.cos(2 * np.pi * train["idx"] / 52)

     features = ["idx", "sin52", "cos52"]
     model = LinearRegression()
     model.fit(train[features], train["vendas"])

     future_idx = np.arange(len(train), len(train) + horizonte)
     future_dates = pd.date_range(start=train.index[-1] + pd.Timedelta(weeks=1), periods=horizonte, freq="W")
     future = pd.DataFrame({
        "vendas": np.nan,
         "idx": future_idx,
        "sin52": np.sin(2 * np.pi * future_idx / 52),
         "cos52": np.cos(2 * np.pi * future_idx / 52)
     }, index=future_dates)
 
     preds = model.predict(future[features])
     future["prev"] = preds
     prev = pd.concat([
         train[["vendas"]].rename(columns={"vendas": "hist"}),
         future[["prev"]]
     ], axis=0)
 
     fig_prev = px.line(prev, title=f"Vendas Semanais – {cat_escolhida} (Histórico x Previsão)")
     st.plotly_chart(fig_prev, use_container_width=True)
 
     # MAPE simples via backtest curto (últimas 8 semanas)
     back_n = min(8, len(train) - 10)
     if back_n > 0:
         hist_train = train.iloc[:-back_n]
         hist_test = train.iloc[-back_n:]

         model_bt = LinearRegression()
         model_bt.fit(hist_train[features], hist_train["vendas"])
         pred_bt = model_bt.predict(hist_test[features])
         denom = hist_test["vendas"].clip(lower=1e-6)
         mape = (np.abs((hist_test["vendas"] - pred_bt) / denom)).replace([np.inf, -np.inf], np.nan).mean() * 100
         st.caption(f"MAPE (backtest {back_n} semanas): {mape:,.1f}%")
 else:
     st.info("Dados insuficientes para previsão semanal desta categoria. Selecione outra ou amplie o período.")
 
 st.divider()
 
 # -----------------------------
 # 7) Explicações rápidas do dashboard
 # -----------------------------
 with st.expander("O que cada KPI/Tabela/Gráfico significa?"):
     st.markdown("""
 **KPIs**
 - **GMROI** = Lucro Bruto / Estoque médio ao custo. Eficácia do capital empatado no estoque.  
 - **Giro de Estoque** = COGS / Estoque médio. Indica velocidade de venda e reposição.  
 - **UPT (Itens/Transação)** = Unidades vendidas / nº de transações. Proxy de “tamanho da cesta”.  
 - **Ticket Médio (R$)** = Vendas / nº de transações. Valor médio por compra.  
 - **OTIF (%)**: % de pedidos entregues no prazo e completos.  
 - **Shrink (%)**: perdas de estoque (furtos, quebras, erros) sobre vendas.  
 
 **Gráficos**
 - **Linha**: tendência (sazonalidade, promoções).  
 - **Barras por Categoria**: foco no mix e na margem.  
 - **Pareto (Top SKUs)**: priorização (80/20).  
 
 **Tabelas**
 - **Top/Bottom SKUs**: ação rápida (precificação, exposição, ruptura).  
 - **Rupturas (proxy)**: alerta por SKU/loja com baixa média e alta variância.  
 
 **Previsibilidade**
-- **Holt-Winters**: capta nível, tendência e sazonalidade semanal.  
+- **Regressão sazonal**: tendência linear com harmônicos semanais (senóides/cossenoides).
 - **MAPE**: erro percentual médio do backtest; quanto menor, melhor.
 """)
