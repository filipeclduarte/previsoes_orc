import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import plotly.io as pio

def generate_pdf_report(df_filtered, df_monthly, fig_temporal, ano):
    buffer_pdf = io.BytesIO()
    c = canvas.Canvas(buffer_pdf, pagesize=letter)
    width, height = letter

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Relatório Orçamentário de Transporte - {ano}")

    # Resumo Estatístico
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Resumo para o ano {ano}:")
    c.drawString(50, height - 100, f"Receita Total: R${df_filtered['Receita_Transporte'].sum():,.2f}")
    c.drawString(50, height - 120, f"Despesa Total: R${df_filtered['Despesa_Transporte'].sum():,.2f}")
    c.drawString(50, height - 140, f"Saldo Total: R${df_filtered['Saldo_Transporte'].sum():,.2f}")

    # Gráfico Temporal
    fig_temporal.update_layout(width=500, height=300)
    img_data = pio.to_image(fig_temporal, format="png")
    img = ImageReader(io.BytesIO(img_data))
    c.drawImage(img, 50, height - 450, width=500, height=300)

    # Insights
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 480, "Insights:")
    c.setFont("Helvetica", 12)
    insights = []
    if df_monthly["Saldo_Transporte"].mean() < 0:
        insights.append("- Saldo médio negativo: Revisar despesas ou aumentar receitas.")
    else:
        insights.append("- Saldo médio positivo: Boa saúde financeira.")
    if df_monthly["Despesa_Transporte"].mean() > df_monthly["Receita_Transporte"].mean() * 0.9:
        insights.append("- Despesas altas (>90% da receita): Otimizar custos.")
    
    for i, insight in enumerate(insights):
        c.drawString(50, height - 500 - i * 20, insight)

    c.showPage()
    c.save()
    buffer_pdf.seek(0)
    return buffer_pdf