"""P1: ZL6 로거 파일 → 구역별 PPFD 시계열(SiteFrame).

`agri-logger-qc` 스킬에서 검증된 ZL6 export 처리 규칙을 그대로 따른다:
  - 시트는 'Processed'(보정값) 우선. Raw 는 정수 카운트라 쓰면 안 된다.
  - 헤더는 1행이 아니다. 'Timestamp' 가 있는 행을 찾아 헤더로 쓴다.
  - 여러 export 파일 병합은 **컬럼 합집합**으로. 첫 파일 스키마로 고정하면
    중간에 추가된 센서 데이터가 통째로 사라진다 (실제로 발생했던 데이터 손실 버그).
  - timestamp 는 완전한 10분 격자로 채우고, 빠진 시각은 빈 행으로 삽입한다.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SiteConfig

TIMESTAMP_KEYS = ("timestamp", "일시", "datetime", "date_time", "시간")


# =====================================================================
# 파일 읽기
# =====================================================================

def _pick_sheet(sheet_names: list[str]) -> str:
    """ZL6 export 시트 선택: Processed(보정값) > Metadata 아닌 첫 시트."""
    for s in sheet_names:
        if "processed" in str(s).lower():
            return s
    non_meta = [s for s in sheet_names if "metadata" not in str(s).lower()]
    return non_meta[0] if non_meta else sheet_names[0]


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 20) -> int:
    """'Timestamp' 류 셀이 있는 행을 헤더로 본다. 단일헤더 파일은 0행에서 바로 잡힌다."""
    for i in range(min(max_scan, len(raw))):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist() if c is not None]
        if any(any(k in c for k in TIMESTAMP_KEYS) for c in cells):
            return i
    return 0


def _unique_columns(cells: list) -> list[str]:
    cols, seen = [], {}
    for i, c in enumerate(cells):
        name = "" if c is None else str(c).strip()
        if name == "" or name.lower() == "nan":
            name = f"Unnamed_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    return cols


def read_table(buffer, filename: str) -> pd.DataFrame:
    """파일 경로 또는 업로드된 file-like 객체를 DataFrame 으로 읽는다.

    Streamlit 업로드와 로컬 경로가 같은 경로를 타도록 buffer 를 받는다.
    dtype=object 로 읽어 '#VALUE!' 같은 오류 문자열을 보존한다.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(buffer, engine="openpyxl" if ext == ".xlsx" else None)
        raw = xl.parse(sheet_name=_pick_sheet(xl.sheet_names), header=None, dtype=object)
    elif ext == ".csv":
        data = buffer.read() if hasattr(buffer, "read") else Path(buffer).read_bytes()
        raw = None
        for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
            try:
                raw = pd.read_csv(io.BytesIO(data), header=None, dtype=object, encoding=enc)
                break
            except Exception:
                continue
        if raw is None:
            raise ValueError(f"CSV 인코딩을 인식할 수 없습니다: {filename}")
    else:
        raise ValueError(f"지원하지 않는 형식입니다: {filename} (csv/xlsx 만 지원)")

    if raw is None or raw.dropna(how="all").empty:
        raise ValueError(f"'{filename}' 에서 읽을 수 있는 데이터가 없습니다.")

    hdr = _detect_header_row(raw)
    df = raw.iloc[hdr + 1:].copy()
    df.columns = _unique_columns(raw.iloc[hdr].tolist())
    return df.dropna(how="all").reset_index(drop=True)


def read_logger_file(path: str | Path) -> pd.DataFrame:
    """단일 로거 파일(.xlsx/.csv)을 DataFrame 으로."""
    path = Path(path)
    return read_table(path, path.name)


def find_column(df: pd.DataFrame, keyword: str, exclude: tuple[str, ...] = ()) -> str | None:
    """부분일치(대소문자 무시)로 컬럼을 찾는다. 정확일치를 우선한다."""
    kw = keyword.strip().lower()
    cols = list(df.columns)
    for c in cols:
        if str(c).strip().lower() == kw:
            return c
    for c in cols:
        low = str(c).lower()
        if kw in low and not any(x in low for x in exclude):
            return c
    return None


def detect_timestamp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if any(k in str(c).strip().lower() for k in TIMESTAMP_KEYS):
            return c
    # 키워드로 못 찾으면 첫 컬럼이 날짜로 파싱되는지 확인
    first = df.columns[0]
    parsed = pd.to_datetime(df[first], errors="coerce")
    if parsed.notna().mean() > 0.8:
        return first
    raise ValueError(f"timestamp 컬럼을 찾지 못했습니다. 컬럼: {list(df.columns)[:8]}")


def to_numeric(series: pd.Series) -> pd.Series:
    """'#VALUE!' 등 오류 토큰은 NaN 으로. 결측(NaN)과 오류를 구분하지 않고 모두 결측 처리한다.

    (열 제거 판단은 P1 범위 밖 — 여기서는 값 단위로만 정리한다.)
    """
    return pd.to_numeric(series, errors="coerce")


def logger_id_of(filename: str) -> str:
    """'z6-21068_061225-0906.xlsx' → 'z6-21068'. 첫 언더스코어 앞을 그룹 키로 쓴다.

    로거ID에 하이픈은 있어도 언더스코어는 없다는 ZL6 파일명 규칙에 의존한다.
    (숫자를 떼는 방식은 시리얼까지 지워서 서로 다른 로거를 합쳐버린다.)
    """
    stem = Path(filename).stem
    return stem.split("_")[0]


# =====================================================================
# 병합 / 격자화
# =====================================================================

def merge_logger_files(paths: list[str | Path]) -> pd.DataFrame:
    """같은 로거의 여러 export 를 컬럼 **합집합**으로 병합한다."""
    frames = []
    for p in paths:
        df = read_logger_file(p)
        ts_col = detect_timestamp_column(df)
        df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        frames.append(df.dropna(subset=["timestamp"]))
    if not frames:
        raise ValueError("병합할 파일이 없습니다.")
    # concat 은 컬럼 합집합으로 동작한다 (없는 컬럼은 NaN) — 이게 핵심.
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # 겹치는 기간은 나중 파일 우선
    merged = merged.drop_duplicates(subset="timestamp", keep="last")
    return merged.sort_values("timestamp").reset_index(drop=True)


def build_grid(series_map: dict[str, pd.Series], interval_minutes: int
               ) -> tuple[pd.DataFrame, dict]:
    """여러 구역 시계열을 하나의 완전한 시간 격자 위에 정렬한다.

    - 부동소수점 드리프트(19:59:59.995)는 가장 가까운 격자로 반올림한다. 안 하면
      외부 데이터와 timestamp 가 안 맞아 전부 결측이 된다.
    - 빠진 시각은 빈 행으로 삽입한다.
    """
    freq = f"{interval_minutes}min"
    normalized, report = {}, {"drift_fixed": 0, "duplicates_dropped": 0}
    for name, s in series_map.items():
        idx = pd.to_datetime(pd.Series(s.index)).dt.round(freq)
        report["drift_fixed"] += int((idx.values != pd.to_datetime(pd.Series(s.index)).values).sum())
        t = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(idx))
        before = len(t)
        t = t[~t.index.duplicated(keep="first")]
        report["duplicates_dropped"] += before - len(t)
        normalized[name] = t

    starts = [t.index.min() for t in normalized.values() if len(t)]
    ends = [t.index.max() for t in normalized.values() if len(t)]
    if not starts:
        raise ValueError("유효한 데이터가 없습니다.")
    grid = pd.date_range(min(starts), max(ends), freq=freq)

    out = pd.DataFrame(index=grid)
    out.index.name = "timestamp"
    for name, t in normalized.items():
        out[name] = t.reindex(grid)

    report.update({
        "rows": len(out),
        "start": grid.min(),
        "end": grid.max(),
        "inserted_rows": int(len(grid) - max((len(t) for t in normalized.values()), default=0)),
    })
    return out, report


# =====================================================================
# SiteFrame
# =====================================================================

@dataclass
class SiteFrame:
    """사이트 전체의 시간 정렬된 PPFD 데이터.

    ppfd    : index=timestamp, columns=zone_id  (µmol·m⁻²·s⁻¹)
    external: index=timestamp, 외부(온실 밖) PPFD
    source  : ppfd 와 같은 shape. 각 값의 출처 플래그.
              measured / estimated_sibling / estimated_external / estimated_night / missing
    """

    ppfd: pd.DataFrame
    external: pd.Series
    source: pd.DataFrame
    interval_minutes: int
    report: dict
    temperature: pd.DataFrame | None = None   # 안전 조건(고온 차단)용. 없으면 None.

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.ppfd.index

    @property
    def zone_ids(self) -> list[str]:
        return list(self.ppfd.columns)

    @property
    def interval_seconds(self) -> int:
        return self.interval_minutes * 60

    def missing_ratio(self) -> pd.Series:
        return (self.source == "missing").mean()

    def measured_ratio(self) -> pd.Series:
        return (self.source == "measured").mean()

    @property
    def has_temperature(self) -> bool:
        return self.temperature is not None and not self.temperature.empty

    def slice(self, start=None, end=None) -> "SiteFrame":
        """[start, end) 구간을 잘라낸 새 SiteFrame."""
        idx = self.ppfd.index
        mask = np.ones(len(idx), dtype=bool)
        if start is not None:
            mask &= idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx < pd.Timestamp(end)
        temp = self.temperature.loc[mask] if self.has_temperature else None
        return SiteFrame(self.ppfd.loc[mask], self.external.loc[mask], self.source.loc[mask],
                         self.interval_minutes, self.report, temp)

    @classmethod
    def from_wide(cls, ppfd: pd.DataFrame, external: pd.Series, interval_minutes: int,
                  report: dict | None = None,
                  temperature: pd.DataFrame | None = None) -> "SiteFrame":
        """이미 정리된 wide 프레임으로 SiteFrame 을 만든다 (테스트·합성데이터용)."""
        ppfd = ppfd.sort_index()
        external = external.reindex(ppfd.index)
        source = pd.DataFrame(
            np.where(ppfd.notna(), "measured", "missing"),
            index=ppfd.index, columns=ppfd.columns)
        if temperature is not None:
            temperature = temperature.reindex(ppfd.index)
        return cls(ppfd, external, source, interval_minutes, report or {}, temperature)


def discover_logger_files(logger_dir: Path, logger_id: str) -> list[Path]:
    """logger_id 를 포함하는 파일들을 찾는다 (파일명 규칙: <로거ID>_<내려받은일시>)."""
    if not logger_dir.exists():
        return []
    hits = []
    for p in sorted(logger_dir.iterdir()):
        if p.suffix.lower() not in (".xlsx", ".xls", ".csv"):
            continue
        if logger_id and (logger_id in p.name or logger_id_of(p.name) == logger_id):
            hits.append(p)
    return hits


def load_external_ppfd(external_dir: Path, keyword: str) -> pd.Series:
    """외부 PPFD 파일들을 병합해 하나의 시계열로. 겹치는 구간은 나중 파일 우선."""
    if not external_dir.exists():
        return pd.Series(dtype=float)
    paths = [p for p in sorted(external_dir.iterdir())
             if p.suffix.lower() in (".xlsx", ".xls", ".csv")]
    if not paths:
        return pd.Series(dtype=float)
    merged = merge_logger_files(paths)
    col = find_column(merged, keyword) or find_column(merged, "ppfd")
    if col is None:
        raise ValueError(
            f"외부 데이터에서 '{keyword}' 컬럼을 찾지 못했습니다. 컬럼: {list(merged.columns)[:10]}")
    s = pd.Series(to_numeric(merged[col]).to_numpy(),
                  index=pd.DatetimeIndex(merged["timestamp"]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_site_frame(cfg: SiteConfig, logger_dir: Path | None = None,
                    external_dir: Path | None = None) -> SiteFrame:
    """site.yaml 의 구역 매핑에 따라 로거 파일들을 읽어 SiteFrame 을 만든다."""
    logger_dir = Path(logger_dir) if logger_dir else cfg.logger_dir
    external_dir = Path(external_dir) if external_dir else cfg.external_dir

    series_map: dict[str, pd.Series] = {}
    per_zone_report: dict[str, dict] = {}
    for zone in cfg.zones:
        paths = discover_logger_files(logger_dir, zone.logger_id)
        if not paths:
            raise FileNotFoundError(
                f"구역 {zone.id!r}(로거 {zone.logger_id!r})의 파일을 {logger_dir} 에서 찾지 못했습니다.")
        merged = merge_logger_files(paths)
        col = find_column(merged, zone.ppfd_column)
        if col is None:
            raise ValueError(
                f"구역 {zone.id!r} 파일에서 '{zone.ppfd_column}' 컬럼을 찾지 못했습니다. "
                f"컬럼: {list(merged.columns)[:10]}")
        series_map[zone.id] = pd.Series(to_numeric(merged[col]).to_numpy(),
                                        index=pd.DatetimeIndex(merged["timestamp"]))
        per_zone_report[zone.id] = {"files": [p.name for p in paths], "column": col}

    external = load_external_ppfd(external_dir, cfg.external.ppfd_column)
    if len(external):
        series_map["__external__"] = external

    grid, report = build_grid(series_map, cfg.interval_minutes)
    ext_series = (grid.pop("__external__") if "__external__" in grid.columns
                  else pd.Series(np.nan, index=grid.index))
    report["zones"] = per_zone_report
    if not len(external):
        report["warning"] = (
            f"외부 PPFD 데이터를 {external_dir} 에서 찾지 못했습니다. "
            "투과율 보정과 청천지수 계산이 내부 데이터만으로 이뤄집니다.")

    return SiteFrame.from_wide(grid[[z.id for z in cfg.zones]], ext_series,
                               cfg.interval_minutes, report)


# =====================================================================
# 업로드 파일 처리 (Streamlit 등)
# =====================================================================

def numeric_columns(df: pd.DataFrame, ts_col: str | None = None) -> list[str]:
    """숫자로 해석 가능한 컬럼만 (측정값 후보)."""
    out = []
    for c in df.columns:
        if ts_col is not None and c == ts_col:
            continue
        vals = to_numeric(df[c])
        if vals.notna().mean() > 0.5:
            out.append(c)
    return out


def guess_ppfd_columns(columns: list[str]) -> list[str]:
    """'B5_5_PPFD', ' µmol·m⁻²·s⁻¹ PPFD' 처럼 온실 **내부** PPFD 로 보이는 컬럼을 추린다.

    외부(온실 밖) PPFD 는 구역이 아니라 기준값이므로 제외한다 — 구역으로 잡히면
    투과율 100% 인 가짜 구역이 하나 생긴다.
    """
    keys = ("ppfd", "µmol", "umol", "광량", "quantum", "par")
    outside = ("외부", "external", "outdoor", "옥외", "노지", "기상", "퀀텀센터")
    hits = [c for c in columns
            if any(k in str(c).lower() for k in keys)
            and not any(x in str(c).lower() for x in outside)]
    return hits or list(columns)


def guess_external_column(columns: list[str]) -> str | None:
    """외부(온실 밖) PPFD 컬럼 추정."""
    outside = ("외부", "external", "outdoor", "옥외", "노지", "퀀텀센터")
    for c in columns:
        low = str(c).lower()
        if any(x in low for x in outside):
            return c
    return None


def guess_temperature_column(columns: list[str]) -> str | None:
    """기온 컬럼 추정. 지온·배지온도는 제외해야 한다 (둘 다 'temperature' 를 포함)."""
    exclude = ("soil", "substrate", "배지", "근권", "water", "logger", "device")
    for c in columns:
        low = str(c).lower()
        if any(k in low for k in ("air temperature", "기온", "대기온도")):
            return c
    for c in columns:
        low = str(c).lower()
        if ("temp" in low or "온도" in low) and not any(x in low for x in exclude):
            return c
    return None


def build_frame_from_columns(df: pd.DataFrame, ts_col: str, zone_columns: dict[str, str],
                             interval_minutes: int, external_column: str | None = None,
                             temperature_column: str | None = None) -> SiteFrame:
    """사용자가 고른 컬럼 매핑으로 SiteFrame 을 만든다.

    zone_columns: {zone_id: 원본 컬럼명}
    """
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    valid = ts.notna()
    if valid.sum() == 0:
        raise ValueError(f"'{ts_col}' 컬럼을 시각으로 해석하지 못했습니다.")
    index = pd.DatetimeIndex(ts[valid])

    series_map = {zid: pd.Series(to_numeric(df.loc[valid, col]).to_numpy(), index=index)
                  for zid, col in zone_columns.items()}
    if external_column:
        series_map["__external__"] = pd.Series(
            to_numeric(df.loc[valid, external_column]).to_numpy(), index=index)
    if temperature_column:
        series_map["__temperature__"] = pd.Series(
            to_numeric(df.loc[valid, temperature_column]).to_numpy(), index=index)

    grid, report = build_grid(series_map, interval_minutes)
    ext = (grid.pop("__external__") if "__external__" in grid.columns
           else pd.Series(np.nan, index=grid.index))
    temp = grid.pop("__temperature__") if "__temperature__" in grid.columns else None
    zone_ids = list(zone_columns)
    temp_df = (pd.DataFrame({z: temp for z in zone_ids}, index=grid.index)
               if temp is not None else None)
    report["columns"] = dict(zone_columns)
    return SiteFrame.from_wide(grid[zone_ids], ext, interval_minutes, report, temp_df)
