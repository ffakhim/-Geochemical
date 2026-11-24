import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, linregress, skew, kurtosis
from matplotlib.backends.backend_pdf import PdfPages

# غیرفعال کردن هشدارهای غیرضروری برای تمیز ماندن خروجی
warnings.filterwarnings("ignore")

# --- تنظیمات و ثابت‌ها ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "geochemical_data.csv"
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# عناصر مورد نظر برای تحلیل (باید دقیقاً با نام ستون‌های فایل CSV مطابقت داشته باشد)
INTERESTING_ELEMENTS = ["Cu(%)", "As(%)", "Ag(%)", "S(%)", "Fe(%)", "Al(%)", "Si(%)"]
DEPTH_COL = "depth(m)"

# رنگ‌ها برای یکدستی نمودارها
COLORS = {
    "PRIMARY": "#157A8A", # مسی‌رنگ
    "SECONDARY": "#B22222", # قرمز
    "GOLD": "#C58A1B",
    "BACKGROUND": "#F0F0F0"
}

CHART_SETTINGS = {
    "figsize": (9, 7),
    "dpi": 150,
    "font_family": "DejaVu Sans"
}

# -----------------------------------------------------------------------------
# توابع کمکی
# -----------------------------------------------------------------------------

def load_and_prepare_data(file_path):
    """
    فایل CSV را بارگذاری می‌کند و همه ستون‌ها را به عددی تبدیل می‌کند،
    مقادیر '<LOD' را به NaN تبدیل می‌کند.
    """
    print(f"Loading data from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    data = pd.read_csv(file_path)
    
    # تبدیل همه ستون‌ها به عددی. 'coerce' خطاها (مانند '<LOD') را به 'NaN' تبدیل می‌کند
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    
    return data

def calculate_descriptive_statistics(data, elements):
    """
    آمار توصیفی را برای لیست داده شده از عناصر محاسبه می‌کند.
    """
    print("Calculating statistics...")
    summary_rows = []

    for col in elements:
        if col in data.columns:
            values = data[col].dropna()
            if len(values) == 0:
                continue
                
            row = {
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "CV_%": (values.std(ddof=1) / values.mean()) * 100 if values.mean() != 0 else 0,
                "min": values.min(),
                "25%": values.quantile(0.25),
                "median": values.median(),
                "75%": values.quantile(0.75),
                "95%": values.quantile(0.95),
                "max": values.max(),
                "skewness": skew(values, bias=False),
                "kurtosis": kurtosis(values, bias=False)
            }
            summary_rows.append(row)

    stats_df = pd.DataFrame(summary_rows, index=[col for col in elements if col in data.columns])
    return stats_df

# -----------------------------------------------------------------------------
# توابع مصورسازی
# -----------------------------------------------------------------------------

def create_depth_profiles(data, elements):
    """
    نمودارهای پروفیل عمق را برای عناصر مشخص شده ایجاد می‌کند.
    """
    print("Creating depth profile charts...")
    # فیلتر کردن فقط عناصری که در داده وجود دارند
    valid_elements = [el for el in elements if el in data.columns]
    if not valid_elements:
        return

    num_elements = len(valid_elements)
    cols = 2 if num_elements > 1 else 1
    rows = (num_elements + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 6*rows), squeeze=False)
    
    for i, el in enumerate(valid_elements):
        ax = axes[i//cols, i%cols]
        ax.plot(data[el], data[DEPTH_COL], color=COLORS["PRIMARY"], marker="o", markersize=3)
        ax.set(xlabel=f"{el}", ylabel="Depth (m)", title=f"{el} depth profile")
        ax.invert_yaxis()  # عمق به سمت پایین افزایش می‌یابد
        ax.grid(alpha=0.3)

    # پنهان کردن محورهای استفاده نشده
    for j in range(i + 1, rows * cols):
        axes[j//cols, j%cols].axis('off')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_depth_profiles.png", dpi=CHART_SETTINGS["dpi"])
    plt.show()  # نمایش نمودار روی صفحه
    plt.close(fig)

def create_element_histograms(data, elements):
    """
    هیستوگرام‌هایی برای عناصر مشخص شده با استفاده از مقیاس لگاریتمی ایجاد می‌کند.
    """
    print("Creating element distribution histograms...")
    valid_elements = [el for el in elements if el in data.columns]
    if not valid_elements:
        return

    num_elements = len(valid_elements)
    cols = 2
    rows = (num_elements + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows))
    if num_elements == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, el in enumerate(valid_elements):
        ax = axes[i]
        values = data[el].dropna()
        if len(values) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(f"{el}")
            continue

        ax.hist(values, bins=30, color=COLORS["GOLD"], alpha=0.6, edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel(f"{el} (log scale)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{el} distribution")

    # پنهان کردن محورهای استفاده نشده
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_histograms.png", dpi=CHART_SETTINGS["dpi"])
    plt.show()  # نمایش نمودار روی صفحه
    plt.close(fig)

def create_correlation_heatmap(data, elements):
    """
    نقشه حرارتی همبستگی را برای عناصر مشخص شده ایجاد می‌کند.
    """
    print("Creating correlation heatmap...")
    valid_elements = [el for el in elements if el in data.columns]
    if len(valid_elements) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = data[valid_elements].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlOrBr", vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Element correlation matrix")
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_correlation_heatmap.png", dpi=CHART_SETTINGS["dpi"])
    plt.show()  # نمایش نمودار روی صفحه
    plt.close(fig)

def create_element_vs_element_plot(data, el1, el2):
    """
    نمودار رگرسیون بین دو عنصر را ایجاد می‌کند.
    """
    if el1 not in data.columns or el2 not in data.columns:
        return
    print(f"Creating {el1} vs {el2} regression plot...")
    
    # هم‌تراز کردن داده‌ها با حذف ردیف‌هایی که یکی از عناصر NaN است
    common = data[[el1, el2]].dropna()
    if len(common) < 2:
        return
        
    x = common[el2]
    y = common[el1]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(x, y, alpha=0.6, color=COLORS["PRIMARY"], s=30)
    
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color=COLORS["SECONDARY"], linewidth=2)
    
    ax.set(xlabel=f"{el2}", ylabel=f"{el1}", title=f"{el1} vs {el2}")
    ax.text(0.05, 0.95, f"R² = {r_value**2:.3f}\n{el1} = {intercept:.3f} + {slope:.3f} * {el2}", 
            transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.8))
    
    fig.tight_layout()
    safe_el1 = el1.replace('(','_').replace(')','_').replace('%','_pct').replace('*','_star')
    safe_el2 = el2.replace('(','_').replace(')','_').replace('%','_pct').replace('*','_star')
    fig.savefig(OUTPUT_DIR / f"fig4_{safe_el1}_vs_{safe_el2}.png", dpi=CHART_SETTINGS["dpi"])
    plt.show()  # نمایش نمودار روی صفحه
    plt.close(fig)

def detect_and_plot_anomalies(data, element):
    """
    ناهنجاری‌ها را در یک عنصر با استفاده از قاعده آماری (میانگین + ۲ انحراف معیار از مقادیر log10) شناسایی می‌کند.
    """
    if element not in data.columns:
        return None, 0
    print(f"Detecting anomalies for {element}...")
    
    vals = data[element].dropna()
    if len(vals) == 0:
        return None, 0
    
    log_vals = np.log10(vals)
    threshold = log_vals.mean() + 2 * log_vals.std(ddof=1)
    is_anomalous = log_vals > threshold
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(data.loc[vals.index, DEPTH_COL], log_vals, color="#555", label="log10(value)")
    ax.axhline(threshold, color=COLORS["SECONDARY"], linestyle="--", label="Threshold (Mean + 2SD)")
    
    anomalous_indices = is_anomalous[is_anomalous].index
    anomalous_data = data.loc[anomalous_indices]
    if not anomalous_data.empty:
        ax.scatter(anomalous_data[DEPTH_COL], np.log10(anomalous_data[element]), color="#D62728", s=35, label="Anomaly")
    
    ax.set(xlabel="Depth (m)", ylabel=f"log10({element})", title=f"{element} anomaly detection")
    ax.legend()
    fig.tight_layout()
    safe_el = element.replace('(','_').replace(')','_').replace('%','_pct').replace('*','_star')
    fig.savefig(OUTPUT_DIR / f"fig6_{safe_el}_anomaly.png", dpi=CHART_SETTINGS["dpi"])
    plt.show()  # نمایش نمودار روی صفحه
    plt.close(fig)
    
    return threshold, is_anomalous.sum()

def create_pdf_report(data, stats_df, anomalies_summary):
    """
    گزارش نهایی PDF شامل آمار خلاصه و تمام نمودارهای تولید شده را ایجاد می‌کند.
    """
    print("Generating final PDF report...")
    report_path = BASE_DIR / "geochemical_analysis_report.pdf"
    
    with PdfPages(report_path) as pdf:
        # --- صفحه ۱: خلاصه ---
        fig_title = plt.figure(figsize=(8.27, 11.69)) # A4
        fig_title.text(0.1, 0.9, "Geochemical Data Analysis Report", fontsize=24, fontweight="bold")
        fig_title.text(0.1, 0.85, "Summary of key elements and statistical distributions", fontsize=14, color="#555")
        
        # جدول آمار توصیفی
        stats_text = "### Descriptive Statistics ###\n\n" + stats_df.to_string()
        fig_title.text(0.1, 0.65, stats_text, fontsize=9, family='monospace', va='top', wrap=True)
        
        # بخش ناهنجاری‌ها
        anomaly_text = "\n\n### Detected Anomalies (log10 Threshold) ###\n"
        for el, count in anomalies_summary.items():
            anomaly_text += f"- {el}: {count} samples detected\n"
        fig_title.text(0.1, 0.35, anomaly_text, fontsize=12, va='top')
        
        fig_title.text(0.1, 0.05, "Generated by Geochemical Python Analysis Tool", fontsize=8, color="#999")
        
        pdf.savefig(fig_title)
        plt.close(fig_title)

        # --- صفحات ۲ به بعد: نمودارها ---
        # مرتب‌سازی برای داشتن ترتیب در PDF
        images = sorted(OUTPUT_DIR.glob("*.png"))
        for img_path in images:
            fig_img = plt.figure(figsize=(8.27, 11.69))
            try:
                img_data = plt.imread(img_path)
                ax = fig_img.add_axes([0.1, 0.15, 0.8, 0.75])
                ax.imshow(img_data)
                ax.axis("off")
                fig_img.text(0.1, 0.9, img_path.name, fontsize=12, fontweight="bold")
                pdf.savefig(fig_img)
            except Exception as e:
                print(f"Could not add image {img_path.name} to report: {e}")
            finally:
                plt.close(fig_img)

    print(f"Report saved to: {report_path}")

# -----------------------------------------------------------------------------
# حلقه اجرای اصلی
# -----------------------------------------------------------------------------

def main():
    try:
        # ۱. بارگذاری و آماده‌سازی
        data = load_and_prepare_data(INPUT_FILE)
        
        # ۲. آمار توصیفی
        stats_df = calculate_descriptive_statistics(data, INTERESTING_ELEMENTS)
        stats_df.to_csv(BASE_DIR / "summary_stats.csv")
        print("\n--- DESCRIPTIVE STATISTICS ---")
        print(stats_df.to_string(float_format=lambda v: f"{v:.4f}"))

        # ۳. مصورسازی
        create_depth_profiles(data, INTERESTING_ELEMENTS[:4])
        create_element_histograms(data, INTERESTING_ELEMENTS[:4])
        create_correlation_heatmap(data, INTERESTING_ELEMENTS)
        
        # رگرسیون بین دو عنصر کلیدی (مثلاً مس و آرسنیک)
        create_element_vs_element_plot(data, "Cu(%)", "As(%)")
        
        # شناسایی ناهنجاری برای برخی عناصر
        anomalies_summary = {}
        for el in INTERESTING_ELEMENTS[:3]:
            res = detect_and_plot_anomalies(data, el)
            if res and res[0] is not None:
                anomalies_summary[el] = res[1]

        # ۴. گزارش PDF
        create_pdf_report(data, stats_df, anomalies_summary)
        
        print("\nSUCCESS: Analysis complete. Check the 'plots' folder and 'geochemical_analysis_report.pdf'.")

    except Exception as e:
        print(f"\nERROR: An unexpected problem occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()