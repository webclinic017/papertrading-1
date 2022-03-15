from trader import *
import json
import logging

with open(ROOT_PATH + "/papertrading/static/ref/sector_map.json", "r") as f:
        SECTOR_MAP = json.load(f)
        STOCKS = list(SECTOR_MAP.keys())

with open(ROOT_PATH + "/papertrading/static/ref/nifty50.json", "r") as f:
        NIFTY_STOCK = json.load(f)

with open(ROOT_PATH + "/papertrading/static/ref/sector_weight.json", "r") as f:
        SECTOR_WEIGHT = json.load(f)

with open(ROOT_PATH + "/papertrading/static/ref/stock_weight.json", "r") as f:
        STOCK_WEIGHT = json.load(f)

SECTORS = ["NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY FMCG", "NIFTY MEDIA",
            "NIFTY METAL", "NIFTY AUTO", "NIFTY ENERGY"]

def get_data(end, nifty, slist=None):
    if end is None:
        end = str(datetime.now().date())

    stock = slist or (NIFTY_STOCK if nifty else STOCKS)
    df = fetch_data(stock, f"{end} 09:00:00", f"{end} 16:00:00", span="30minute")

    vz = {}

    for s in stock:
        try:
            adf = df[df["tradingsymbol"] == s].reset_index(drop=True)
            if adf.shape[0] == 0:
                continue
            values_zones, mpm, xticks, yticks, ymax, ymin, BASE = calculate_value_zones(adf)
            vz[s] = values_zones[-1]
        except Exception as e:
            logging.error(s)
            logging.exception(e)
    return stock, df, vz

def rscore_screener(vz):
    rscore = []
    for s in vz:
        rs, rsmax, rsmin = vz[s]["rscore"], vz[s]["rscore-max"], vz[s]["rscore-min"]
        rscore.append([
            s,
            rs,
            rsmin,
            rsmax,
            rsmax - rs,
            rs - rsmin,
            vz[s]["uext"],
            vz[s]["dext"],
            vz[s]["otype"],
            vz[s]["orange"],
            vz[s]["margin_up"],
            vz[s]["margin_down"],
            vz[s]["tpo-ratio"],
            SECTOR_MAP.get(s, "other").upper(),
            round((1+vz[s]["margin_up"])/(1+vz[s]["margin_down"]), 2)
        ])
    return rscore

def screener_data(end=None, nifty=False):
    stock, df, vz = get_data(end, nifty)
    rscore = rscore_screener(vz)
    return {"rscore": rscore}

def format_record(vz):
    rec = []
    for s in vz:
        rec.append([
            s,
            vz[s]["margin_up"],
            vz[s]["margin_down"],
            vz[s]["uext"],
            vz[s]["dext"],
            f"{vz[s]['rscore']} ({vz[s]['rscore-min']}-{vz[s]['rscore-max']})",
            round((1+vz[s]["margin_up"])/(1+vz[s]["margin_down"]), 2),
            SECTOR_WEIGHT.get(s, 0)
        ])
    return rec

def format_record_2(vz):
    rec = []
    for s in vz:
        rec.append([
            s,
            vz[s]["margin_up"],
            vz[s]["margin_down"],
            vz[s]["uext"],
            vz[s]["dext"],
            f"{vz[s]['rscore']} ({vz[s]['rscore-min']}-{vz[s]['rscore-max']})",
            vz[s]["orange"],
            vz[s]["otype"],
            SECTOR_MAP.get(s, "other").upper(),
            round((1+vz[s]["margin_up"])/(1+vz[s]["margin_down"]), 2)
        ])
    return rec


def extended_sector(vz):
    vz = dict([(s, vz[s]) for s in vz])
    return format_record(vz)

def extended_stock(vz):
    vz = dict([(s, vz[s]) for s in vz if vz[s]["uext"] + vz[s]["dext"] > 0])
    return format_record(vz)

def extended_stock_down(vz):
    vz = dict([(s, vz[s]) for s in vz
                if vz[s]["dext"] > 0 and vz[s]["uext"] == 0 and
                round((1+vz[s]["margin_down"])/(1+vz[s]["margin_up"]), 2) >= 2])
    return format_record(vz)

def extended_stock_up(vz):
    vz = dict([(s, vz[s]) for s in vz
                if vz[s]["uext"] > 0 and vz[s]["dext"] == 0 and
                round((1+vz[s]["margin_up"])/(1+vz[s]["margin_down"]), 2) >= 2])
    return format_record(vz)

def no_extended_stock_screen(vz):
    vz = dict([(s, vz[s]) for s in vz if (vz[s]["uext"] + vz[s]["dext"] == 0) or (vz[s]["uext"] == -1000)])
    return format_record_2(vz)

def events_screener(end=None, nifty=True):
    stock, df, vz = get_data(end, nifty)
    sector, sdf, svz = get_data(end, nifty, slist=SECTORS)

    e_sector = extended_sector(svz)
    e_stock = extended_stock(vz)
    e_stock_up = extended_stock_up(vz)
    e_stock_Down = extended_stock_down(vz)
    market_screeners = rscore_screener(vz)
    no_extended_stock = no_extended_stock_screen(vz)

    return {"extended_sector": e_sector,
            "extended_stock": e_stock,
            "extended_stock_up": e_stock_up,
            "extended_stock_down": e_stock_Down,
            "market_screeners": market_screeners,
            "no_extended_stock": no_extended_stock}
