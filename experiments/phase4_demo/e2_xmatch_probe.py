#!/usr/bin/env python
"""E2 bounded probe: labeled Chen+2020 RRL -> same-star ZTF g+r oids via IRSA TAP.

Confirms path A end-to-end: (1) pull a few labeled RRL (position+period) from
Chen+2020 on VizieR; (2) for each, crossmatch to ztf_objects_dr23 in zg AND zr
within 1.5" -> retrieve the g/r oids + epoch counts. Bounded: <=4 RRL x 2 bands.
"""
import io, csv, sys, time, urllib.parse, urllib.request

VIZ = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
IRSA = "https://irsa.ipac.caltech.edu/TAP/sync"


def tap(url, adql, timeout=30):
    data = urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql
    }).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"User-Agent": "ftp-e2-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8", "replace"))))


# A few labeled RRL: 2 from field 786 (clean halo), 2 from field 486 (low-lat).
boxes = {
    786: (141.628, 153.574, 51.520, 58.380),
    486: (270.910, 277.792, 1.120, 7.980),
}
rrl = []
for fld, (r0, r1, d0, d1) in boxes.items():
    q = ('SELECT TOP 2 RAJ2000,DEJ2000,Per,Type,gmag,rmag,Ng,Nr '
         'FROM "J/ApJS/249/18/table2" WHERE Type LIKE \'RR%%\' '
         'AND RAJ2000 BETWEEN %g AND %g AND DEJ2000 BETWEEN %g AND %g '
         'AND Ng>50 AND Nr>50 ORDER BY rmag' % (r0, r1, d0, d1))
    try:
        rows = tap(VIZ, q)
    except Exception as e:
        print("  [viz] field %d query failed: %s" % (fld, str(e)[:120]))
        continue
    for row in rows:
        row["field"] = fld
        rrl.append(row)
        print("RRL fld=%d %s %s  ra=%s dec=%s  Per=%s  g=%s r=%s  Ng=%s Nr=%s"
              % (fld, row["Type"], "", row["RAJ2000"], row["DEJ2000"],
                 row["Per"], row["gmag"], row["rmag"], row["Ng"], row["Nr"]))

def sep_arcsec(ra, dec, ra2, dec2):
    import math
    cd = math.cos(math.radians(dec))
    return math.hypot((ra - ra2) * cd, dec - dec2) * 3600.0


print("\n=== same-star g+r crossmatch to ztf_objects_dr23 (1.5 arcsec) ===")
for row in rrl:
    ra = float(row["RAJ2000"]); dec = float(row["DEJ2000"]); fld = row["field"]
    got = {}
    for fc in ("zg", "zr"):
        q = ("SELECT oid,filtercode,ngoodobsrel,medianmag,ra,dec "
             "FROM ztf_objects_dr23 WHERE filtercode='%s' AND "
             "CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',%f,%f,0.000417))=1"
             % (fc, ra, dec))
        try:
            rows = tap(IRSA, q, timeout=35)
        except Exception as e:
            print("  [irsa] %.5f %.5f %s failed: %s" % (ra, dec, fc, str(e)[:100]))
            got[fc] = None
            continue
        for o in rows:
            o["sep"] = sep_arcsec(ra, dec, float(o["ra"]), float(o["dec"]))
            o["in_field"] = o["oid"].startswith(str(fld))
        # prefer target-field oid, then highest epoch count
        rows.sort(key=lambda o: (not o["in_field"], -int(o["ngoodobsrel"])))
        got[fc] = rows[0] if rows else None
    print("RRL fld=%s ra=%.5f dec=%.5f Per=%s :" % (fld, ra, dec, row["Per"]))
    for lab, fc in (("g", "zg"), ("r", "zr")):
        o = got.get(fc)
        if o:
            print("    %s: oid=%s nobs=%s medmag=%s sep=%.2f\" in_field=%s"
                  % (lab, o["oid"], o["ngoodobsrel"], o["medianmag"],
                     o["sep"], o["in_field"]))
        else:
            print("    %s: NO MATCH" % lab)
    time.sleep(0.4)
print("\nDONE")
