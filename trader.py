import os
import sys
from copy import deepcopy
from pathlib import Path
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
    data = get_data(bk, start=start, end=end, span=span, names=[symbol], predef=None)
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


def x_axis_market_profile(one, tol=0.001):
    ulimit, llimit = one["high"].max(), one["low"].min()
    spread = round((ulimit + llimit)/2 * tol, 2) # 0.1% of mean price
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



def get_historic_graph(name, start, end):
    start, end = adjust_date(start, end)
    print(start, end)
    df = fetch_data(name, start, end, span="30minute")
    last_close = df["close"].values[-1]

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
            mpm += mp
            BASE += (m_len + GAP)
            xticks.append(BASE)
            values_zones.append(vzones)
    values_zones = open_type_annotation(values_zones, skip=3)

    yticks = [_[1] for _ in mpm]
    ymax, ymin = max(yticks), min(yticks)
    yticks = int(1000 * (ymax - ymin)/ymin)

    fig, ax = plt.subplots(1, 1, figsize=(int(BASE*0.2), int(yticks*0.3)))

    # Plot MARKET PROFILE
    um = set([f"${_[2]}$" for _ in mpm])
    for m in um:
        sset = [_ for _ in mpm if _[2] == m[1]]
        ax.scatter(x=[_[0] for _ in sset], y=[_[1] for _ in sset], marker=m, s=60)

    # Plot POCs and Value Zones
    for vz in values_zones:
        ax.add_patch( Rectangle((vz["x"], vz["y"]), vz["width"], vz["height"], alpha=0.1, color="green") )
        ax.add_patch( Rectangle((vz["x"], vz["poc-price"]), vz["width"], vz["poc-price"]*0.0001, alpha=0.8, color="red") )

    # Plot OPEN-TYPE
    for vz in values_zones:
        if "open-type" in vz:
            ax.text(vz["x"], vz["dhigh"]*1.002, vz["open-type"], style='normal', color="black",
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 4})

    # beautify
    ax.grid()
    _ = plt.xticks(ticks=xticks)
    _ = plt.yticks(ticks=np.arange(ymin, ymax, last_close*0.002))

    rstr = [str(x) for x in list(range(10))]
    shuffle(rstr)
    img_name = f"{name}-{''.join(rstr)}"
    img_name = f"./static/temp/{img_name}.jpg"
    _ = [os.remove(fl) for fl in glob("./static/temp/*")]
    fig.savefig(img_name, bbox_inches='tight')
    fig.close()
    return img_name[1:], last_close


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
        df = df[leadhr*60:]
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
