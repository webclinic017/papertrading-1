import os
import sys
from copy import deepcopy
from pathlib import Path
from PIL import Image
from tick import render_graph

ROOT_PATH = "/".join(os.path.abspath(__file__).split("/")[:-2])
print(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "moneyplantv3"))

from moneyplantv3.common.broker import Broker
from tlib.dataloader import get_data

conf = {"DEFAULT.api_key": "cvhbtz7tf7qtpke0",
        "DEFAULT.login_url": "https://kite.trade/connect/login?api_key=cvhbtz7tf7qtpke0&v=3",
        "DEFAULT.api_secret": "jfdout6qwevtuidn8d8v0iospn11kcbd"
        }
bk = Broker(conf=conf)
kite = bk.get_kite()


from datetime import datetime, timedelta
import numpy as np
from random import random, shuffle
from glob import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
matplotlib.pyplot.switch_backend('Agg')


def s2o(dte):
    # 0 - monday, 6-monday
    return datetime.strptime(dte, '%a %b %d %Y %H:%M:%S %Z%z')

def fetch_data(symbol, start, end, span="minute"):
    data = get_data(bk, start=start, end=end, span=span, names=[symbol] if type(symbol) == str else symbol, predef=None)
    data["datetime"] = [x.replace(tzinfo=None) for x in data["datetime"]]
    data["day"] = [x.date() for x in data["datetime"]]
    data["std"] = (data["high"] - data["low"])/4
    data["mean"] = (data["high"] + data["low"])/2
    return data

def calculate_poc(mp, base=0, vz_tol=0.65):
    mpdf = pd.DataFrame(data=mp, columns=["x", "y", "t"])
    mpdf = mpdf.groupby("y").count().reset_index(drop=False).sort_values("y", ascending=False).reset_index(drop=True)
    max_time = mpdf["x"].max()
    candidates = mpdf[mpdf["x"] == max_time].index.tolist()

    # given multiple equally likely pocs, choose the center one
    if len(candidates) > 1:
        mid_candidate = candidates[int(np.ceil(len(candidates)/2))]
        candidates = sorted(candidates, key=lambda x: x%mid_candidate, reverse=True)

    vzones = {"poc": -1, "vz-upper": 0, "vz-lower": 1e+6}

    for candidate in candidates:
        mpdf = mpdf.reset_index(drop=False)
        mpdf = mpdf.rename(columns={"index": "key"})
        mpdf["key"] -= candidate
        mpdf["key"] = abs(mpdf["key"])
        magg = mpdf.groupby("key").agg({"x": "sum"})
        magg["x"] = magg["x"].cumsum()
        magg["x"] /= magg["x"].values[-1]
        width = magg[magg["x"] > vz_tol].index[0]

        vzu, vzl = max(0, candidate-width), min(candidate+width, mpdf.shape[0]-1)

        if abs(vzones["vz-upper"]-vzones["vz-lower"]) > abs(vzu-vzl):

            vzones = {"poc": candidate,  # POCs and value zone
                      "poc-price": mpdf.loc[candidate, "y"],
                      "vz-upper": vzu,
                      "vz-upper-price": mpdf.loc[vzu, "y"],
                      "vz-lower": vzl,
                      "vz-lower-price": mpdf.loc[vzl, "y"],

                       # daily price stats for open price/range analysis
                      "dhigh": mpdf["y"].values[0],
                      "dlow": mpdf["y"].values[-1],

                      "x": base,  # value zone rectangle cordinates for matplotlib
                      "y": mpdf.loc[vzl, "y"],
                      "width": mpdf.loc[candidate, "x"],
                      "height": abs(mpdf.loc[vzl, "y"] - mpdf.loc[vzu, "y"])}
        del mpdf["key"]
    return vzones

def find_range(xval, low, high):
    lindex, hindex = 0, len(xval)-1
    elow, ehigh = 0, 0
    for i in range(len(xval)):
        if xval[i] <= low:
            lindex = i
        else:
            elow = 1

        if xval[len(xval)-1-i] >= high:
            hindex = len(xval)-1-i
        else:
            ehigh = 1

        if elow and ehigh:
            break
    return lindex, hindex

def rotation_score(spread, mp):
    try:
        mpt = pd.DataFrame(mp, columns=["x", "price", "mark"])
        mpt.loc[mpt["mark"].isin(["O"]), "mark"] = "A"
        if mpt["mark"].unique().shape[0] < 3:
            return -1000, -1000, -1000

        mpt = mpt.groupby("mark").agg({"price": ["max", "min"]}).reset_index(drop=False)
        mpt.columns = ["mark", "max", "min"]
        mpt = mpt.sort_values("mark")
        mpt["pmax"] = mpt["max"].shift(1)
        mpt["pmin"] = mpt["min"].shift(1)
        mpt = mpt.dropna()
        mpt["dmax"] = (mpt["max"] - mpt["pmax"])
        mpt["dmin"] = (mpt["min"] - mpt["pmin"])
        mpt["hscore"] = [0 if abs(mpt.loc[i, "dmax"]) < spread else abs(mpt.loc[i, "dmax"])/mpt.loc[i, "dmax"] for i in mpt.index]
        mpt["lscore"] = [0 if abs(mpt.loc[i, "dmin"]) < spread else abs(mpt.loc[i, "dmin"])/mpt.loc[i, "dmin"] for i in mpt.index]
        mpt["score"] = mpt["hscore"] + mpt["lscore"]
        mpt["cscore"] = mpt["score"].cumsum()
    except Exception as e:
        return -1000, -1000, -1000
    return mpt["cscore"].values[-1], mpt["cscore"].min(), mpt["cscore"].max()

def extension_score(mp):
    mpt = pd.DataFrame(mp, columns=["x", "price", "mark"])
    mpt.loc[mpt["mark"].isin(["O", "B"]), "mark"] = "A"
    if mpt["mark"].unique().shape[0] < 3:
        return -1000, -1000

    mpt = mpt.groupby("mark").agg({"price": ["max", "min"]}).reset_index(drop=False)
    mpt.columns = ["mark", "max", "min"]
    mpt = mpt.sort_values("mark")

    def get_crange(series, maxv=True):
        a = series[0]
        new_series = [a]
        for i in series[1:]:
            if (maxv and i > a) | ((not maxv) and i < a):
                a = i
            new_series.append(a)
        return new_series
    mpt["rh"] = get_crange(mpt["max"], maxv=True)
    mpt["rl"] = get_crange(mpt["min"], maxv=False)
    mpt["uext"] = mpt["rh"].rolling(2).apply(lambda x: int(x[0] != x[1]), raw=True).fillna(0).cumsum()
    mpt["dext"] = mpt["rl"].rolling(2).apply(lambda x: int(x[0] != x[1]), raw=True).fillna(0).cumsum()
    return mpt["uext"].values[-1], mpt["dext"].values[-1]


def x_axis_market_profile(one, tol=0.001):
    ulimit, llimit = one["high"].max(), one["low"].min()
    # spread = round((ulimit + llimit)/2 * tol, 2) # 0.1% of mean price
    spread = round(one.loc[0, "open"] * tol, 2)
    xval = np.arange(llimit, ulimit, spread).tolist()
    xval.append(xval[-1]+spread)
    return xval, spread

def x_axis_subset(xval, subone):
    ulimit, llimit = subone["high"].max(), subone["low"].min()
    lindex, hindex = find_range(xval, llimit, ulimit)
    return xval[lindex:hindex+1]

def market_profile(one, spread, xval=None, base=0):
    ulimit, llimit = one["high"].max(), one["low"].min()
    xval = np.array(xval)
    mpm = np.zeros((xval.shape[0], one.shape[0]))

    for i in one.index:
        true_index = find_range(xval, one.loc[i, "low"], one.loc[i, "high"])
        true_index = list(range(true_index[0], true_index[1]+1))
        mpm[true_index, i] = 65 + (i%26)

        if i == 0:
            oindex = np.where(xval <= one.loc[i, "open"] )[0].tolist()[-1]
            mpm[oindex, i] = 79

    mdf = pd.DataFrame(data=[(a, b, mpm[a, b]) for a, b in zip(*np.where(mpm > 0))], columns=["x", "y", "v"])
    mdf = mdf.groupby("x").agg({"v": list}).reset_index(drop=False).sort_values("x")["v"].tolist()
    max_length = max(len(_) for _ in mdf)
    mpl = []

    for k, row in enumerate(mdf):
        mpl += list([[base + x, xval[k], chr(int(v)) ] for x, v in enumerate(row)])
    vzones = calculate_poc(mpl, base)
    vzones["open-price"] = one.loc[0, "open"]
    return mpl, max_length, vzones

def open_type_annotation(vzones, skip=1):
    for i in range(0, len(vzones))[::skip]:
        if i == 0:
            continue

        op = vzones[i]["open-price"]
        if op > vzones[i-1]["vz-lower-price"] and op < vzones[i-1]["vz-upper-price"]:
            vzones[i]["open-type"] = "IN-VALUE"
        elif op > vzones[i-1]["dlow"] and op < vzones[i-1]["dhigh"]:
            vzones[i]["open-type"] = "IN-RANGE"
        else:
            vzones[i]["open-type"] = "OUTSIDE"
    return vzones

def adjust_date(start, end):
    ostart = deepcopy(start)
    oend = deepcopy(end)
    days = abs((start - end).days)
    rdays = 1

    while rdays < days:
        end -= timedelta(days = 1)
        if not end.weekday() in [5, 6]:
            rdays += 1
    return end, oend

def data(name, start, end):
    start, end = adjust_date(start, end)
    df = fetch_data(name, start, end, span="30minute")
    last_close = df["close"].values[-1]
    return df, last_close

def calc_opentype(mp):
    mpt = pd.DataFrame(mp, columns=["x", "price", "mark"])
    mpt = mpt[mpt["mark"].isin(["O", "A", "B"])]
    open = mpt.loc[mpt["mark"] == "O", "price"].values[0]
    above = 1 + mpt[mpt["price"] > open].shape[0]
    below = 1 + mpt[mpt["price"] < open].shape[0]
    range = mpt["price"].unique().shape[0]
    return round(abs(above/below), 4), range

def calc_tporatio(mp, vzones):
    mpt = pd.DataFrame(mp, columns=["x", "price", "mark"])
    poc = vzones["poc-price"]

    onetpo = mpt.groupby("price").agg({"mark": "count"}).reset_index(drop=False)
    onetpo = onetpo.loc[onetpo["mark"] == 1, "price"].to_list()
    mpt = mpt[~mpt["price"].isin(onetpo)]
    uptpo = 1+mpt[mpt["price"] > poc].shape[0]
    downtpo = 1+mpt[mpt["price"] < poc].shape[0]
    return round(downtpo/uptpo, 3)



def calculate_value_zones(df):
    BASE = 0
    GAP = 1
    mpm = []
    xticks = [0]
    values_zones = []

    for k, day in enumerate(df["day"].unique()):
        one = df[df["day"] == day].reset_index(drop=True)
        # calculte x-axis from final day price
        xval, spread = x_axis_market_profile(one, tol=0.001)

        for subone in [one[:2], one[:4], one]:  # 1 hour, 2 hours, and entire day
            # select a subset of x-axis for part day
            subxval = x_axis_subset(xval, subone)
            mp, m_len, vzones = market_profile(subone, spread, xval=subxval, base=BASE)
            vzones["rscore"], vzones["rscore-min"], vzones["rscore-max"] = rotation_score(spread, mp)
            vzones["uext"], vzones["dext"] = extension_score(mp)
            vzones["otype"], vzones["orange"] = calc_opentype(mp)
            vzones["tpo-ratio"] = calc_tporatio(mp, vzones)
            mpm += mp
            BASE += (m_len + GAP)
            xticks.append(BASE)
            values_zones.append(vzones)
    values_zones = open_type_annotation(values_zones, skip=3)

    # Y-TICKS
    yticks = [_[1] for _ in mpm]
    ymax, ymin = max(yticks), min(yticks)
    yticks = int(1000 * (ymax - ymin)/ymin)

    return values_zones, mpm, xticks, yticks, ymax, ymin, BASE


def render_market_profile(ax, mpm):
    # Plot MARKET PROFILE
    um = set([f"${_[2]}$" for _ in mpm])
    for m in um:
        sset = [_ for _ in mpm if _[2] == m[1]]
        ax.scatter(x=[_[0] for _ in sset], y=[_[1] for _ in sset], marker=m, s=60)


def render_last_price(ax, xticks, last_close):
    ax.scatter(x=[xticks[-1]+1], y=[last_close], marker="<", s=60)


def render_pocs(ax, values_zones):
    # Plot POCs and Value Zones
    for vz in values_zones:
        ax.add_patch( Rectangle((vz["x"], vz["y"]), vz["width"], vz["height"], alpha=0.1, color="green") )
        ax.add_patch( Rectangle((vz["x"], vz["poc-price"]), vz["width"], vz["poc-price"]*0.0001, alpha=0.8, color="red") )


def render_open_type(ax, values_zones):
    # Plot OPEN-TYPE
    ofclolor = {"IN-VALUE": "lime", "IN-RANGE": "yellow", "OUTSIDE": "orangered"}
    for vz in values_zones:
        if "open-type" in vz:
            fcolor = ofclolor[vz["open-type"]]
            ax.text(vz["x"], vz["dhigh"]*1.002, vz["open-type"], style='normal', color="black", size="x-large",
                bbox={'facecolor': fcolor, 'alpha': 0.8, 'pad': 4})

def render_rscore(ax, values_zones):
    for vz in values_zones[2::3]:
        if "rscore" in vz and vz["rscore"] != -1000:
            fcolor = "yellow" if abs(vz["rscore"]) < 3 else ("lime" if vz["rscore"] >= 3 else "orangered")
            ax.text(vz["x"]-5, vz["dlow"]*0.996, f"Rotation {vz['rscore']} ({vz['rscore-min']} - {vz['rscore-max']})", style='normal', color="black", size="x-large",
                bbox={'facecolor': fcolor, 'alpha': 0.8, 'pad': 4})

def render_extension(ax, values_zones):
    for vz in values_zones[2::3]:
        if "uext" in vz and vz["uext"] != -1000:
            fcolor = "yellow" if abs(vz["uext"] - vz["dext"]) < 2 else ("lime" if vz["uext"] > vz["dext"] else "orangered")

            ax.text(vz["x"]-5, vz["dlow"]*0.994, f"UE {vz['uext']} | DE {vz['dext']} | TPO {vz['tpo-ratio']}", style='normal', color="black", size="x-large",
                bbox={'facecolor': fcolor, 'alpha': 0.8, 'pad': 4})


def beautify_graph(name, ax, xticks, ymin, ymax, last_close):
    # beautify
    ax.grid()
    ax.set_title(name)
    _ = plt.xticks(ticks=xticks)
    _ = plt.yticks(ticks=np.arange(ymin, ymax, last_close*0.002))


def save_image(fig, name):
    rstr = [str(x) for x in list(range(10))]
    shuffle(rstr)
    img_name = f"{name}-{''.join(rstr)}"
    img_name = f"./static/temp/{img_name}.jpg"
    _ = [os.remove(fl) for fl in glob(f"./static/temp/{name}-*")]
    fig.savefig(img_name, bbox_inches='tight')
    plt.close(fig)
    return img_name


def get_historic_graph(name, start, end):
    df, last_close = data(name, start, end)
    values_zones, mpm, xticks, yticks, ymax, ymin, BASE = calculate_value_zones(df)

    fig, ax = plt.subplots(1, 1, figsize=(int(BASE*0.2), int(yticks*0.3)))
    render_market_profile(ax, mpm)
    render_last_price(ax, xticks, last_close)
    render_pocs(ax, values_zones)
    render_open_type(ax, values_zones)
    render_rscore(ax, values_zones)
    render_extension(ax, values_zones)
    beautify_graph(f"{name} | {str(start)} | {str(end)}", ax, xticks, ymin, ymax, last_close)
    img_name = save_image(fig, name)
    return img_name[1:], last_close


def get_volume_data(nm, end, lead=None):
    img = render_graph(nm, end, lead)
    im = Image.fromarray(img)

    _ = [os.remove(fl) for fl in glob(f"./static/temp/{nm}-tick-*")]
    rstr = [str(x) for x in list(range(10))]
    shuffle(rstr)
    img_name = f"{nm}-tick-{''.join(rstr)}"
    img_name = f"./static/temp/{img_name}.jpg"
    im.save(img_name)
    return [(img_name[1:], 0)]


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.ones(shape=(max_height, total_width, 3))*255
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output.astype(np.uint8)

def merge_output(outcome):
    if len(outcome) == 1:
        return outcome

    images = ["."+oc[0] for oc in outcome]
    last_closes = [oc[1] for oc in outcome][0]
    output = concat_n_images(images)
    im = Image.fromarray(output)

    name = "-".join([x.split("/")[-1].split(".")[0].split("-")[0] for x in images])
    _ = [os.remove(fl) for fl in glob(f"./static/temp/{name}-*")]
    rstr = [str(x) for x in list(range(10))]
    shuffle(rstr)
    img_name = f"{name}-{''.join(rstr)}"
    img_name = f"./static/temp/{img_name}.jpg"
    im.save(img_name)

    return [(img_name, last_closes)]

def format_payload(name, start, intraday, entry, stoploss, target, buy, gameid, change, partial):
    document = {
        "tradingsymbol": name,
        "entry_time": start,
        "intraday": intraday,
        "entry": entry,
        "stoploss": stoploss,
        "target": target,
        "buy": buy,
        "game-id": gameid,
        "change": change,
        "partial": partial
    }
    return document


def bet_outcome(name, start, fwd, intraday, leadhr, entry, stoploss, target, buy):
    if (buy and not (stoploss < entry < target)) or ((not buy) and not (stoploss > entry > target)):
        return -1000, False

    if intraday:
        df = fetch_data(name, start, start+timedelta(hours=17), span="minute")
        df = df[leadhr*30:]
        close = df["close"].values
        # print(close)

        outcome = None
        active = False
        if buy:
            for value in close:
                if value <= entry:
                    active = True

                if active and value >= target:
                    outcome = 100 * (target - entry)/entry, False
                elif active and value <= stoploss:
                    outcome = 100 * (stoploss - entry)/entry, False
            if active and (not outcome):
                outcome = 100 * (value - entry)/entry, True
            elif not active:
                outcome = 0, True
        else:
            for value in close:
                if value >= entry:
                    active = True

                if active and value <= target:
                    outcome = 100 * (entry - target)/entry, False
                elif active and value >= stoploss:
                    outcome = 100 * (entry - stoploss)/entry, False
            if active and (not outcome):
                outcome = 100 * (entry - value)/entry, True
            elif not active:
                outcome = 0, True
        return outcome
    else:
        return -1000, False
