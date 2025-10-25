import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression

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
    saz = 1.0 + 0.15 * np.sin(2 * np.pi * (d.timetuple().tm_yday) / 365)  # sazonalidade anual suave
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
                rows.append([
                    d,
                    loja,
                    canal,
                    sku,
                    vendas,
                    qtd,
                    margem,
                    cogs,
                    estoque_custo,
                    promo,
                    otif_ok,
                    transacoes,
                ])

df = pd.DataFrame(
    rows,
    columns=[
        "data",
        "loja",
        "canal",
        "sku",
        "vendas",
        "qtd",
        "margem",
        "cogs",
        "estoque_custo",
        "promo",
        "otif_ok",
        "transacoes",
    ],
).merge(catalogo, on="sku", how="left")

# Métrica de shrink fictícia (1.3% média, por ruído)
df["shrink_valor"] = df["vendas"] * np.clip(np.random.normal(0.013, 0.004, size=len(df)), 0, 0.05)

# -----------------------------
# 2) Seleção de página
# -----------------------------
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Escolha a página",
    ("Visão Comercial", "Churn & Cohort"),
)

if pagina == "Visão Comercial":
    # -----------------------------
    # 2A) Barra lateral – filtros
    # -----------------------------
    st.sidebar.title("Filtros")
    periodo = st.sidebar.date_input("Período", [dates.min(), dates.max()])
    loja_sel = st.sidebar.multiselect("Loja", lojas, default=lojas)
    canal_sel = st.sidebar.multiselect("Canal", canais, default=canais)
    cat_sel = st.sidebar.multiselect("Categoria", cats, default=cats)

    mask = (
        (df["data"].between(pd.to_datetime(periodo[0]), pd.to_datetime(periodo[1])))
        & (df["loja"].isin(loja_sel))
        & (df["canal"].isin(canal_sel))
        & (df["categoria"].isin(cat_sel))
    )
    dff = df.loc[mask].copy()

    # -----------------------------
    # 3) KPIs
    # -----------------------------
    receita = dff["vendas"].sum()
    margem_val = dff["margem"].sum()
    margem_pct = (margem_val / receita) if receita else 0

    # GMROI = Lucro Bruto / Estoque médio ao custo (proxy simples: média da coluna estoque_custo)
    gmroi = margem_val / max(1.0, dff["estoque_custo"].mean())

    # Giro = COGS / Estoque médio ao custo (proxy)
    giro = dff["cogs"].sum() / max(1.0, dff["estoque_custo"].mean())

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
    c2.metric("Margem Bruta (%)", f"{100 * margem_pct:,.1f}%".replace(",", "."))
    c3.metric("GMROI", f"{gmroi:,.2f}".replace(",", "."))
    c4.metric("Giro de Estoque", f"{giro:,.2f}".replace(",", "."))
    c5.metric("UPT (itens/trans.)", f"{upt:,.2f}".replace(",", "."))
    c6.metric("OTIF (%)", f"{otif:,.1f}%".replace(",", "."))

    c7, c8 = st.columns(2)
    c7.metric("Ticket Médio (R$)", f"{ticket_medio:,.2f}".replace(",", "."))
    c8.metric("Shrink (%)", f"{100 * shrink_pct:,.2f}%".replace(",", "."))

    st.divider()

    # -----------------------------
    # 4) Gráficos principais
    # -----------------------------
    # Tendência vendas e margem
    serie = dff.groupby("data").agg(vendas=("vendas", "sum"), margem=("margem", "sum")).reset_index()
    fig_linha = px.line(serie, x="data", y=["vendas", "margem"], title="Tendência: Vendas e Margem (R$)")
    st.plotly_chart(fig_linha, use_container_width=True)

    # Barras por categoria
    cat_view = dff.groupby("categoria").agg(vendas=("vendas", "sum"), margem=("margem", "sum")).reset_index()
    fig_cat = px.bar(
        cat_view.sort_values("vendas", ascending=False),
        x="categoria",
        y=["vendas", "margem"],
        barmode="group",
        title="Vendas e Margem por Categoria",
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Pareto SKUs (Top 20 por vendas)
    sku_view = dff.groupby(["sku", "categoria"]).agg(vendas=("vendas", "sum"), margem=("margem", "sum")).reset_index()
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
    rupt = (
        dff.groupby(["loja", "sku"])
        .agg(qtd_med=("qtd", "mean"), qtd_cv=("qtd", lambda x: x.std() / x.mean() if x.mean() else np.nan))
        .reset_index()
    )
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

    serie_cat = (
        dff.loc[dff["categoria"] == cat_escolhida]
        .groupby("data")["vendas"].sum()
        .resample("W").sum()
    )

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
        future = pd.DataFrame(
            {
                "vendas": np.nan,
                "idx": future_idx,
                "sin52": np.sin(2 * np.pi * future_idx / 52),
                "cos52": np.cos(2 * np.pi * future_idx / 52),
            },
            index=future_dates,
        )

        preds = model.predict(future[features])
        future["prev"] = preds
        prev = pd.concat([
            train[["vendas"]].rename(columns={"vendas": "hist"}),
            future[["prev"]],
        ])

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
            mape = (
                np.abs((hist_test["vendas"] - pred_bt) / denom)
                .replace([np.inf, -np.inf], np.nan)
                .mean()
                * 100
            )
            st.caption(f"MAPE (backtest {back_n} semanas): {mape:,.1f}%")
    else:
        st.info("Dados insuficientes para previsão semanal desta categoria. Selecione outra ou amplie o período.")

    st.divider()

    # -----------------------------
    # 7) Explicações rápidas do dashboard
    # -----------------------------
    with st.expander("O que cada KPI/Tabela/Gráfico significa?"):
        st.markdown(
            """
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
- **Regressão sazonal**: tendência linear com harmônicos semanais (senóides/cossenoides).
- **MAPE**: erro percentual médio do backtest; quanto menor, melhor.
"""
        )
else:
    # -----------------------------
    # Página de Churn & Cohort
    # -----------------------------
    st.title("Churn & Cohort – Fidelização de Clientes")

    # Base fictícia mensal de churn
    meses_clientes = pd.date_range("2023-01-01", periods=24, freq="MS")
    clientes_ativos = 1200
    churn_rows = []
    for mes in meses_clientes:
        novos = np.random.poisson(80)
        churnados = int(clientes_ativos * np.random.uniform(0.04, 0.085))
        churn_rate = churnados / max(clientes_ativos, 1)
        clientes_finais = clientes_ativos + novos - churnados
        churn_rows.append(
            {
                "mes": mes,
                "clientes_inicio": clientes_ativos,
                "novos_clientes": novos,
                "clientes_churn": churnados,
                "clientes_fim": max(clientes_finais, 300),
                "churn_rate": churn_rate,
            }
        )
        clientes_ativos = max(clientes_finais, 300)

    df_churn = pd.DataFrame(churn_rows)

    media_churn = 100 * df_churn["churn_rate"].mean()
    churn_ultimo = 100 * df_churn.iloc[-1]["churn_rate"]
    clientes_base_atual = int(df_churn.iloc[-1]["clientes_fim"])

    k1, k2, k3 = st.columns(3)
    k1.metric("Churn Médio (24m)", f"{media_churn:,.2f}%".replace(",", "."))
    k2.metric("Churn Último Mês", f"{churn_ultimo:,.2f}%".replace(",", "."))
    k3.metric("Base Atual de Clientes", f"{clientes_base_atual:,.0f}".replace(",", "."))

    st.divider()

    fig_churn = px.line(
        df_churn,
        x="mes",
        y=["clientes_inicio", "clientes_fim"],
        title="Evolução da Base de Clientes",
        labels={"value": "Clientes", "variable": "Status"},
    )
    fig_churn.update_traces(mode="lines+markers")
    st.plotly_chart(fig_churn, use_container_width=True)

    fig_churn_bar = px.bar(
        df_churn,
        x="mes",
        y=["novos_clientes", "clientes_churn"],
        title="Novos vs Churn por Mês",
        barmode="group",
        labels={"value": "Clientes", "variable": "Movimento"},
    )
    st.plotly_chart(fig_churn_bar, use_container_width=True)

    st.markdown(
        """
**Como interpretar**: acompanhar a tendência do churn versus aquisição ajuda a ajustar ações de fidelização,
campanhas de reativação e balancear investimento em prospecção x retenção.
"""
    )

    st.divider()

    # Cohort fictício
    cohort_meses = pd.period_range("2023-01", "2024-12", freq="M")
    meses_observados = pd.period_range("2023-01", "2025-06", freq="M")
    cohort_registros = []
    for cohort in cohort_meses:
        base = np.random.randint(160, 320)
        sobrevivencia = np.clip(np.linspace(1.0, 0.25, num=min(12, len(meses_observados))), 0.05, 1.0)
        ruido = np.random.uniform(0.85, 1.05, size=survivencia.size)
        for idx, proporcao in enumerate(sobrevivencia):
            periodo = cohort + idx
            if periodo > meses_observados[-1]:
                break
            clientes_ativos = max(int(base * proporcao * ruido[idx]), 5)
            cohort_registros.append(
                {
                    "cohort": cohort.to_timestamp(),
                    "periodo": periodo.to_timestamp(),
                    "meses_apos": idx,
                    "clientes": clientes_ativos,
                    "base_cohort": base,
                }
            )

    df_cohort = pd.DataFrame(cohort_registros)
    df_cohort["retencao_pct"] = df_cohort["clientes"] / df_cohort["base_cohort"]

    tabela_retencao = (
        df_cohort.pivot_table(
            index="cohort",
            columns="meses_apos",
            values="retencao_pct",
            aggfunc="mean",
        )
        .sort_index(ascending=False)
    )

    st.subheader("Retenção por Cohort (12 meses)")
    fig_cohort = px.imshow(
        tabela_retencao.fillna(0),
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Meses após 1ª compra", y="Cohort de Entrada", color="Retenção"),
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

    tabela_retencao_pct = tabela_retencao.applymap(lambda x: f"{100 * x:,.1f}%" if pd.notna(x) else "-")
    st.dataframe(tabela_retencao_pct)

    st.markdown(
        """
**Insights**: o heatmap evidencia a velocidade de queda da retenção por cohort. Busque padrões
de melhora (tons mais escuros) após ações de CRM, programas de fidelidade ou ajustes no mix de ofertas.
"""
    )
