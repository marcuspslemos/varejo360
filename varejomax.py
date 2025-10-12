import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
            for sku in sample_skus:
                base_q = np.random.poisson(5)
                promo = np.random.rand() < 0.12  # 12% dos casos em promoção
                fator_promo = 1.35 if promo else 1.0
                qtd = max(0, int(np.round(base_q * saz * fator_promo + np.random.randn())))
                preco = np.random.uniform(6, 60)
                cogs_unit = preco * np.random.uniform(0.55, 0.8)
                vendas = qtd * preco
                cogs = qtd * cogs_unit
                margem = vendas - cogs
                estoque_custo = np.random.uniform(50, 400)  # proxy para custo médio diário do SKU
                otif_ok = np.random.rand() < 0.93  # 93% pedidos OK como proxy
                transacoes = max(1, int(np.round(qtd / np.random.uniform(1.2, 2.8))))  # aproximação UPT
                rows.append([d, loja, canal, sku, vendas, qtd, margem, cogs, estoque_custo,
                             promo, otif_ok, transacoes])

df = pd.DataFrame(rows, columns=[
    "data","loja","canal","sku","vendas","qtd","margem","cogs","estoque_custo",
    "promo","otif_ok","transacoes"
]).merge(catalogo, on="sku", how="left")

# Métrica de shrink fictícia (1.3% média, por ruído)
df["shrink_valor"] = df["vendas"] * np.clip(np.random.normal(0.013, 0.004, size=len(df)), 0, 0.05)

# -----------------------------
# 2) Barra lateral – filtros
# -----------------------------
st.sidebar.title("Filtros")
periodo = st.sidebar.date_input("Período", [dates.min(), dates.max()])
loja_sel = st.sidebar.multiselect("Loja", lojas, default=lojas)
canal_sel = st.sidebar.multiselect("Canal", canais, default=canais)
cat_sel = st.sidebar.multiselect("Categoria", cats, default=cats)

mask = (
    (df["data"].between(pd.to_datetime(periodo[0]), pd.to_datetime(periodo[1]))) &
    (df["loja"].isin(loja_sel)) &
    (df["canal"].isin(canal_sel)) &
    (df["categoria"].isin(cat_sel))
)
dff = df.loc[mask].copy()

# -----------------------------
# 3) KPIs
# -----------------------------
receita = dff["vendas"].sum()
margem_val = dff["margem"].sum()
margem_pct = (margem_val / receita) if receita else 0

# GMROI = Lucro Bruto / Estoque médio ao custo (proxy simples: média da coluna estoque_custo)
gmroi = (margem_val / max(1.0, dff["estoque_custo"].mean()))

# Giro = COGS / Estoque médio ao custo (proxy)
giro = (dff["cogs"].sum() / max(1.0, dff["estoque_custo"].mean()))

# UPT = unidades / transações
upt = dff["qtd"].sum() / max(1, dff["transacoes"].sum())

# Ticket médio (ABV) = vendas / transações
ticket_medio = receita / max(1, dff["transacoes"].sum())

# Shrink %
shrink_pct = dff["shrink_valor"].sum() / max(1.0, receita)

# OTIF %
otif = 100.0 * (dff["otif_ok"].sum() / len(dff)) if len(dff) else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Receita (R$)", f"{receita:,.0f}".replace(",", "."))
c2.metric("Margem Bruta (%)", f"{100*margem_pct:,.1f}%".replace(",", "."))
c3.metric("GMROI", f"{gmroi:,.2f}".replace(",", "."))
c4.metric("Giro de Estoque", f"{giro:,.2f}".replace(",", "."))
c5.metric("UPT (itens/trans.)", f"{upt:,.2f}".replace(",", "."))
c6.metric("OTIF (%)", f"{otif:,.1f}%".replace(",", "."))

c7, c8 = st.columns(2)
c7.metric("Ticket Médio (R$)", f"{ticket_medio:,.2f}".replace(",", "."))
c8.metric("Shrink (%)", f"{100*shrink_pct:,.2f}%".replace(",", "."))

st.divider()

# -----------------------------
# 4) Gráficos principais
# -----------------------------
# Tendência vendas e margem
serie = dff.groupby("data").agg(vendas=("vendas","sum"), margem=("margem","sum")).reset_index()
fig_linha = px.line(serie, x="data", y=["vendas","margem"], title="Tendência: Vendas e Margem (R$)")
st.plotly_chart(fig_linha, use_container_width=True)

# Barras por categoria
cat_view = dff.groupby("categoria").agg(vendas=("vendas","sum"), margem=("margem","sum")).reset_index()
fig_cat = px.bar(cat_view.sort_values("vendas", ascending=False), x="categoria", y=["vendas","margem"],
                 barmode="group", title="Vendas e Margem por Categoria")
st.plotly_chart(fig_cat, use_container_width=True)

# Pareto SKUs (Top 20 por vendas)
sku_view = dff.groupby(["sku","categoria"]).agg(vendas=("vendas","sum"), margem=("margem","sum")).reset_index()
top_skus = sku_view.sort_values("vendas", ascending=False).head(20)
fig_sku = px.bar(top_skus, x="sku", y="vendas", color="categoria", title="Top 20 SKUs por Vendas (Pareto)")
st.plotly_chart(fig_sku, use_container_width=True)

st.divider()

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
st.header("Previsibilidade (Forecast) – Holt-Winters")
cat_escolhida = st.selectbox("Categoria para prever", options=sorted(dff["categoria"].unique()))
horizonte = st.slider("Horizonte (semanas à frente)", 4, 16, 8)

serie_cat = (dff.loc[dff["categoria"]==cat_escolhida]
                .groupby("data")["vendas"].sum()
                .resample("W").sum())

if len(serie_cat.dropna()) > 20:
    train = serie_cat.copy()
    # Modelo aditivo com tendência e sazonalidade semanal
    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=52, initialization_method="estimated")
    model = hw.fit(optimized=True, use_brute=True)
    fcast = model.forecast(horizonte)
    prev = pd.concat([train.rename("hist"), fcast.rename("prev")], axis=1)

    fig_prev = px.line(prev, title=f"Vendas Semanais – {cat_escolhida} (Histórico x Previsão)")
    st.plotly_chart(fig_prev, use_container_width=True)

    # MAPE simples via backtest curto (últimas 8 semanas)
    back_n = min(8, len(train)-10)
    if back_n > 0:
        hist_train = train.iloc[:-back_n]
        hist_test = train.iloc[-back_n:]
        model_bt = ExponentialSmoothing(hist_train, trend="add", seasonal="add",
                                        seasonal_periods=52, initialization_method="estimated").fit()
        pred_bt = model_bt.forecast(back_n)
        mape = (np.abs((hist_test - pred_bt) / hist_test)).replace([np.inf, -np.inf], np.nan).mean() * 100
        st.caption(f"MAPE (backtest {back_n} semanas): {mape:,.1f}%")
else:
    st.info("Dados insuficientes para previsão semanal desta categoria. Selecione outra ou amplie o período.")

st.subheader("Módulo de Machine Learning – Previsão diária por loja/canal/categoria")
col_ml1, col_ml2, col_ml3 = st.columns(3)
with col_ml1:
    loja_ml = st.selectbox("Loja (ML)", options=sorted(df["loja"].unique()))
with col_ml2:
    canal_ml = st.selectbox("Canal (ML)", options=sorted(df["canal"].unique()))
with col_ml3:
    categoria_ml = st.selectbox("Categoria (ML)", options=sorted(df["categoria"].unique()))

col_ml4, col_ml5 = st.columns(2)
with col_ml4:
    horizonte_ml = st.slider("Horizonte (dias à frente)", 7, 30, 14)
with col_ml5:
    promo_ml = st.slider("Premissa de promoções futuras (%)", 0, 40, 12, step=1) / 100

mask_ml = (
    (df["loja"] == loja_ml) &
    (df["canal"] == canal_ml) &
    (df["categoria"] == categoria_ml)
)

serie_ml = (df.loc[mask_ml]
              .groupby("data")
              .agg(vendas=("vendas", "sum"), promo_rate=("promo", "mean"))
              .reset_index()
              .sort_values("data"))

if not serie_ml.empty:
    serie_ml["promo_rate"] = serie_ml["promo_rate"].fillna(0.0)
    serie_ml["dow"] = serie_ml["data"].dt.dayofweek
    serie_ml["month"] = serie_ml["data"].dt.month
    serie_ml["is_weekend"] = serie_ml["dow"].isin([5, 6]).astype(int)
    serie_ml["trend"] = np.arange(len(serie_ml))
    serie_ml["lag_1"] = serie_ml["vendas"].shift(1)
    serie_ml["lag_7"] = serie_ml["vendas"].shift(7)
    serie_ml["ma_7"] = serie_ml["vendas"].shift(1).rolling(7).mean()

    feature_cols = ["dow", "month", "is_weekend", "trend", "promo_rate", "lag_1", "lag_7", "ma_7"]
    treino_ml = serie_ml.dropna(subset=feature_cols + ["vendas"]).reset_index(drop=True)

    if len(treino_ml) > 30:
        split_idx = int(len(treino_ml) * 0.8)
        treino_df = treino_ml.iloc[:split_idx]
        teste_df = treino_ml.iloc[split_idx:]

        modelo_treino = RandomForestRegressor(n_estimators=400, random_state=42)
        modelo_treino.fit(treino_df[feature_cols], treino_df["vendas"])

        if not teste_df.empty:
            preds_teste = modelo_treino.predict(teste_df[feature_cols])
            mae = mean_absolute_error(teste_df["vendas"], preds_teste)
            rmse = mean_squared_error(teste_df["vendas"], preds_teste, squared=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_vals = np.where(teste_df["vendas"] != 0,
                                     np.abs((teste_df["vendas"] - preds_teste) / teste_df["vendas"]),
                                     np.nan)
            mape_ml = np.nanmean(mape_vals) * 100
        else:
            mae = rmse = mape_ml = np.nan

        modelo = RandomForestRegressor(n_estimators=400, random_state=42)
        modelo.fit(treino_ml[feature_cols], treino_ml["vendas"])

        historico_plot = treino_ml[["data", "vendas"]].copy()
        historico_plot["tipo"] = "Histórico"

        historico_iter = treino_ml.copy()
        previsoes = []

        for passo in range(horizonte_ml):
            prox_data = historico_iter["data"].iloc[-1] + pd.Timedelta(days=1)
            dow = prox_data.dayofweek
            month = prox_data.month
            is_weekend = int(dow >= 5)
            prox_trend = historico_iter["trend"].iloc[-1] + 1
            lag_1 = historico_iter["vendas"].iloc[-1]
            if len(historico_iter) >= 7:
                lag_7 = historico_iter["vendas"].iloc[-7]
            else:
                lag_7 = lag_1
            ma_7 = historico_iter["vendas"].tail(7).mean()

            features_row = pd.DataFrame([{ 
                "dow": dow,
                "month": month,
                "is_weekend": is_weekend,
                "trend": prox_trend,
                "promo_rate": promo_ml,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "ma_7": ma_7
            }])

            prev_val = float(modelo.predict(features_row[feature_cols])[0])
            previsoes.append({"data": prox_data, "previsao": prev_val})

            novo_registro = {
                "data": prox_data,
                "vendas": prev_val,
                "promo_rate": promo_ml,
                "dow": dow,
                "month": month,
                "is_weekend": is_weekend,
                "trend": prox_trend,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "ma_7": ma_7
            }
            historico_iter = pd.concat([historico_iter, pd.DataFrame([novo_registro])], ignore_index=True)

        if previsoes:
            previsoes_df = pd.DataFrame(previsoes)
            previsoes_df["tipo"] = "Previsão"
            previsoes_df = previsoes_df.rename(columns={"previsao": "vendas"})

            plot_ml = pd.concat([historico_plot, previsoes_df], ignore_index=True)
            fig_ml = px.line(plot_ml, x="data", y="vendas", color="tipo", markers=True,
                             title=f"Random Forest – {loja_ml} / {canal_ml} / {categoria_ml}")
            st.plotly_chart(fig_ml, use_container_width=True)

            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric("MAE (R$)", f"{mae:,.0f}".replace(",", ".") if not np.isnan(mae) else "-")
            col_met2.metric("RMSE (R$)", f"{rmse:,.0f}".replace(",", ".") if not np.isnan(rmse) else "-")
            col_met3.metric("MAPE (%)", f"{mape_ml:,.1f}%".replace(",", ".") if not np.isnan(mape_ml) else "-")

            tabela_prev = previsoes_df[["data", "vendas"]].copy()
            tabela_prev["premissa_promo_%"] = promo_ml * 100
            st.dataframe(tabela_prev.rename(columns={"vendas": "vendas_previstas"}))
        else:
            st.info("O modelo treinou, mas não foi possível gerar previsões futuras com os parâmetros atuais.")
    else:
        st.info("Dados históricos insuficientes para treinar o modelo de ML. Amplie o período ou escolha outra combinação.")
else:
    st.info("Não há histórico para a combinação selecionada.")

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
- **Holt-Winters**: capta nível, tendência e sazonalidade semanal.  
- **MAPE**: erro percentual médio do backtest; quanto menor, melhor.
""")
