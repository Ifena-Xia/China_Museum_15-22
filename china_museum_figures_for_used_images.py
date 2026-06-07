"""
Inputs (place beside script): GDP_per_capita.xlsx, region_year.xlsx
Run:  python china_museum_figures.py
Deps: pandas, numpy, matplotlib, scipy, openpyxl
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from scipy.stats import pearsonr, spearmanr

OUT="figs"; os.makedirs(OUT,exist_ok=True); PREFIX="china_museum"
INK="#121212"; MUTED="#6b6b6b"; GRID="#dcdcdc"
ACCENT="#356A8D"      # steel blue, the point color for China Museum
ACCENT2="#9FBCD0"     # light steel
NEUTRAL="#9a9ea3"     # gray baseline
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":INK,
    "axes.linewidth":0.8,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.7,
    "axes.axisbelow":True,"svg.fonttype":"none"})

EAST=["Beijing","Tianjin","Liaoning","Shanghai","Jiangsu","Zhejiang","Fujian","Shandong","Guangdong"]
CENTRAL=["Hebei","Shanxi","Jilin","Heilongjiang","Anhui","Jiangxi","Henan","Hubei","Hunan","Hainan"]
WEST=["Inner Mongolia","Guangxi","Chongqing","Sichuan","Guizhou","Yunnan","Tibet","Shaanxi","Gansu","Qinghai","Ningxia","Xinjiang"]
def region(p):
    if p in EAST: return "East"
    if p in CENTRAL: return "Central"
    if p in WEST: return "West"
    return None

def load():
    gdp=pd.read_excel("GDP_per_capita.xlsx").rename(columns={"Unnamed: 0":"prov"})
    gdp=gdp.melt(id_vars="prov",var_name="year",value_name="gdp_pc")
    xl=pd.ExcelFile("region_year.xlsx"); rows=[]
    for sh in xl.sheet_names:
        d=pd.read_excel("region_year.xlsx",sheet_name=sh).rename(columns={"Unnamed: 0":"prov"})
        d["year"]=int(sh); rows.append(d)
    mus=pd.concat(rows,ignore_index=True)
    mus.columns=[c if c in("prov","year") else c for c in mus.columns]
    mus=mus.rename(columns={"Museum size (10^6 \u33a1)":"size","Subsidy (10^6 RMB)":"subsidy",
                            "Expenditure (10^6 RMB)":"expend","Count":"count"})
    df=mus.merge(gdp,on=["prov","year"],how="inner")
    df["subsidy_sqm"]=df["subsidy"]/df["size"]      # RMB per m2
    df["expend_sqm"]=df["expend"]/df["size"]
    df["region"]=df["prov"].map(region)
    df=df.dropna(subset=["region"])
    for c in ["gdp_pc","subsidy_sqm","expend_sqm","size"]:
        df=df[df[c]>0]
        df["log_"+c]=np.log(df[c])
    return df

def _frame(fig,title,subtitle,source):
    fig.text(0.065,0.965,title,ha="left",va="top",fontsize=15,fontweight="bold",color=INK)
    fig.text(0.065,0.915,subtitle,ha="left",va="top",fontsize=10.3,color=INK)
    fig.text(0.065,0.018,source,ha="left",va="bottom",fontsize=8.5,color=MUTED)
    fig.add_artist(plt.Line2D([0.065,0.115],[0.986,0.986],color=ACCENT,linewidth=3.2,solid_capstyle="butt"))

def verify(df):
    print("=== correlations (pooled province-years, n=%d) ==="%len(df))
    for x,y,lab in [("log_gdp_pc","log_subsidy_sqm","GDP vs subsidy/sqm"),
                    ("log_gdp_pc","log_expend_sqm","GDP vs expenditure/sqm"),
                    ("log_gdp_pc","log_size","GDP vs museum size")]:
        r,p=pearsonr(df[x],df[y]); rs,ps=spearmanr(df[x],df[y])
        print(f"  {lab:26s} Pearson r={r:+.2f} p={p:.3f} | Spearman rho={rs:+.2f} p={ps:.3f}")

def figR(df):
    g=df.groupby(["region","year"]).agg(gdp=("gdp_pc","mean"),sub=("subsidy_sqm","mean"),
                                         exp=("expend_sqm","mean")).reset_index()
    metrics=[("gdp","GDP per capita, RMB"),("sub","Subsidy per m\u00b2, RMB"),("exp","Expenditure per m\u00b2, RMB")]
    colors={"East":ACCENT,"Central":ACCENT2,"West":NEUTRAL}
    fig,axes=plt.subplots(1,3,figsize=(10.8,4.2)); fig.subplots_adjust(left=0.07,right=0.985,top=0.80,bottom=0.20,wspace=0.32)
    for ax,(m,lab) in zip(axes,metrics):
        for reg in ["East","Central","West"]:
            s=g[g["region"]==reg]
            ax.plot(s["year"],s[m],color=colors[reg],linewidth=2.4,marker="o",markersize=3)
            ax.text(s["year"].iloc[-1]+0.05,s[m].iloc[-1],reg,color=colors[reg],fontsize=8.5,va="center",fontweight="bold")
        ax.set_title(lab,fontsize=10.5,fontweight="bold",loc="left",pad=6)
        ax.spines[["top","right"]].set_visible(False); ax.set_xlim(2016,2022.8)
        ax.set_xticks(range(2016,2023,2))
    _frame(fig,"Investment does not track the wealth gap",
        "Regional averages, 2016\u20132022. The East leads on GDP, but subsidy and spending per square meter do not line up the same way.",
        "Source: author's calculations from SYCRC and MCT reports")
    for e in("svg","png"): fig.savefig(f"{OUT}/{PREFIX}_figR_regional.{e}",dpi=200,bbox_inches="tight")
    plt.close(fig)

def figK(df):
    pairs=[("log_subsidy_sqm","Subsidy per m\u00b2 (log)","decoupled"),
           ("log_size","Museum size (log)","tracks wealth")]
    fig,axes=plt.subplots(1,2,figsize=(10.0,4.6)); fig.subplots_adjust(left=0.08,right=0.975,top=0.80,bottom=0.16,wspace=0.22)
    for ax,(y,ylab,tag) in zip(axes,pairs):
        ax.scatter(df["log_gdp_pc"],df[y],s=16,color=ACCENT,alpha=0.55,edgecolor="none")
        b=np.polyfit(df["log_gdp_pc"],df[y],1); xs=np.linspace(df["log_gdp_pc"].min(),df["log_gdp_pc"].max(),50)
        ax.plot(xs,np.polyval(b,xs),color=INK,linewidth=1.6,linestyle=(0,(4,2)))
        r,p=pearsonr(df["log_gdp_pc"],df[y])
        ax.set_title(f"GDP vs {ylab.split(' (')[0].lower()}",fontsize=10.5,fontweight="bold",loc="left",pad=6)
        ax.text(0.04,0.94,f"r = {r:+.2f}"+("  (n.s.)" if p>=0.05 else f"  (p<{0.05})"),transform=ax.transAxes,
                fontsize=10,va="top",color=INK,fontweight="bold")
        ax.set_xlabel("GDP per capita (log)",fontsize=9.5); ax.set_ylabel(ylab,fontsize=9.5)
        ax.spines[["top","right"]].set_visible(False)
    _frame(fig,"Money for buildings, not for running them",
        "Provincial wealth predicts how big museums are, but not how much they are subsidized per square meter. Pooled province-years, 2016\u20132022.",
        "Source: author's calculations from SYCRC and MCT reports")
    for e in("svg","png"): fig.savefig(f"{OUT}/{PREFIX}_figK_keyfinding.{e}",dpi=200,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    df=load(); verify(df); figR(df); figK(df)
    print("Wrote figures with prefix",PREFIX)
