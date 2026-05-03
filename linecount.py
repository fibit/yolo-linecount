from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import ffmpegcv
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO


PREFX = ("rtsp://", "rtmp://", "rtp://", "http://", "https://")  # Network sources
MODLS = Path("models")  # Model files directory
OUTPT = Path("outputs")  # Counter output directory
SNAPS = 60  # Save latest frames every N processed frames
READT = 8.0  # Seconds to wait for a network frame before reconnect
BACKM = 30.0  # Max reconnect backoff in seconds


@dataclass(frozen=True)
class Cnfig:
    model: str
    input: str | int
    scale: int
    linea: tuple[int, int]
    lineb: tuple[int, int]
    clses: list[int] | None
    place: str

    @property
    def outpt(self) -> Path:
        return OUTPT / self.place


def readc(paths: str) -> Cnfig:
    pfile = Path(paths)
    if not pfile.exists():
        raise FileNotFoundError(f"Config file not found: {pfile}")

    with pfile.open("r", encoding="utf-8") as fhand:
        rawjs = json.load(fhand)

    needs = {"model", "source", "resize", "line_start", "line_end", "classes", "output"}
    if not needs.issubset(rawjs):
        misss = needs - rawjs.keys()
        raise ValueError(f"Missing config keys: {', '.join(sorted(misss))}")

    return Cnfig(
        model=rawjs["model"],
        input=rawjs["source"],
        scale=int(rawjs["resize"]),
        linea=tuple(rawjs["line_start"]),
        lineb=tuple(rawjs["line_end"]),
        clses=rawjs["classes"],
        place=rawjs["output"],
    )


class Count:
    def __init__(self, cnfig: Cnfig) -> None:
        self.cnfig = cnfig
        self._mkdir()
        self.isnet = self._isnet(cnfig.input)
        self.msrce = cnfig.model if self._isnet(cnfig.model) else str(MODLS / cnfig.model)
        self.model = self._loadm()
        self.trakr = ByteTrackTracker()
        self.zones = self._linez()
        self.boxan, self.laban, self.linan = self._anots()
        self.captr = self._captu(cnfig.input, cnfig.scale)
        self.framq: queue.Queue[tuple[bool, np.ndarray | None]] = queue.Queue(maxsize=1)
        self.rstop = threading.Event()
        self.rthrd: threading.Thread | None = None
        self.rderr: Exception | None = None

    def runit(self) -> None:
        counr = 0
        backs = 1.0
        sourc = self.cnfig.input
        try:
            if self.isnet:
                self._startr()
            while True:
                try:
                    if self.isnet:
                        ready, frame = self._readt(READT)
                    else:
                        ready, frame = self.captr.read()
                except (TimeoutError, RuntimeError, ValueError) as error:
                    if self.isnet:
                        logging.warning("Capture stalled, reconnecting stream: %s", error)
                        self._reconn(sourc, backs)
                        backs = min(backs * 2.0, BACKM)
                        continue
                    logging.info("Input finished: %s", error)
                    break

                if not ready or frame is None:
                    if not self.isnet:
                        logging.info("Input finished")
                        break
                    logging.warning("Capture returned no frame, reconnecting stream")
                    self._reconn(sourc, backs)
                    backs = min(backs * 2.0, BACKM)
                    continue

                backs = 1.0
                orgnl, annot = self._stepp(frame.copy())
                counr += 1
                if counr >= SNAPS:
                    self._savei("last_original.jpg", orgnl)
                    self._savei("last_annotated.jpg", annot)
                    counr = 0
        finally:
            if self.isnet:
                self._stopr()
            elif self.captr is not None:
                self.captr.release()

    def _startr(self) -> None:
        self._dropq()
        self.rderr = None
        self.rstop = threading.Event()
        self.rthrd = threading.Thread(
            target=self._readr,
            args=(self.captr, self.rstop),
            daemon=True,
        )
        self.rthrd.start()

    def _stopr(self) -> None:
        self.rstop.set()
        if self.captr is not None:
            try:
                self.captr.release()
            except Exception:
                pass
        if self.rthrd is not None:
            self.rthrd.join(timeout=1.0)
            self.rthrd = None

    def _reconn(self, sourc: str | int, backs: float) -> None:
        self._stopr()
        time.sleep(backs)
        self.captr = self._captu(sourc, self.cnfig.scale)
        self._startr()

    def _readt(self, touts: float | None) -> tuple[bool, np.ndarray | None]:
        if self.rderr is not None:
            error = self.rderr
            self.rderr = None
            raise error
        try:
            return self.framq.get(timeout=touts)
        except queue.Empty as error:
            raise TimeoutError(f"Frame read timeout after {touts}s") from error

    def _stepp(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orgnl = frame.copy()
        detss, reslt = self._infer(frame)
        vldet = self._vldid(detss)
        maski, masko = self.zones.trigger(vldet)
        self._logjl(vldet, maski, masko)
        annot = self._drawf(frame, detss, self._label(reslt, detss))
        self._snaps(orgnl, annot, maski, masko)
        return orgnl, annot

    def _infer(self, frame: np.ndarray) -> tuple[sv.Detections, object]:
        reslt = self.model(frame, verbose=False, classes=self.cnfig.clses, imgsz=self.cnfig.scale)[0]
        detss = sv.Detections.from_ultralytics(reslt)
        return self.trakr.update(detss), reslt

    def _label(self, reslt: object, detss: sv.Detections) -> list[str]:
        names = reslt.names
        return [
            f"{names[clsid]} {score:.2f}"
            for clsid, score in zip(detss.class_id, detss.confidence, strict=False)
        ]

    def _drawf(self, frame: np.ndarray, detss: sv.Detections, lblst: list[str]) -> np.ndarray:
        drawn = self.boxan.annotate(frame, detss)
        drawn = self.laban.annotate(drawn, detss, lblst)
        return self.linan.annotate(drawn, self.zones)

    def _snaps(self, orgnl: np.ndarray, annot: np.ndarray, maski: np.ndarray, masko: np.ndarray) -> None:
        if not (np.any(maski) or np.any(masko)):
            return

        if np.any(maski):
            self._savei("last_original_in.jpg", orgnl)
            self._savei("last_annotated_in.jpg", annot)
        if np.any(masko):
            self._savei("last_original_out.jpg", orgnl)
            self._savei("last_annotated_out.jpg", annot)

    def _logjl(self, detss: sv.Detections, maski: np.ndarray, masko: np.ndarray) -> None:
        if not (np.any(maski) or np.any(masko)):
            return

        stamp = datetime.now().isoformat(timespec="milliseconds")

        for index in np.flatnonzero(maski):
            trkid = self._trkid(detss, int(index))
            if trkid is None:
                continue
            self._addjl({"timestamp": stamp, "direction": "in", "track_id": trkid})
        for index in np.flatnonzero(masko):
            trkid = self._trkid(detss, int(index))
            if trkid is None:
                continue
            self._addjl({"timestamp": stamp, "direction": "out", "track_id": trkid})

    def _addjl(self, rowjs: dict) -> None:
        linee = json.dumps(rowjs, ensure_ascii=False) + "\n"
        with self._dayfp().open("a", encoding="utf-8") as fhand:
            fhand.write(linee)

    def _savei(self, filen: str, frame: np.ndarray) -> None:
        cv2.imwrite(str(self.cnfig.outpt / filen), frame)

    def _readr(self, captr: object, rstop: threading.Event) -> None:
        while not rstop.is_set():
            try:
                ready, frame = captr.read()
            except Exception as error:
                if rstop.is_set():
                    break
                self.rderr = error
                ready, frame = False, None

            if rstop.is_set():
                break

            if self.framq.full():
                try:
                    self.framq.get_nowait()
                except queue.Empty:
                    pass
            self.framq.put((ready, frame))

            if not ready:
                break

    def _dropq(self) -> None:
        while True:
            try:
                self.framq.get_nowait()
            except queue.Empty:
                break

    def _mkdir(self) -> None:
        MODLS.mkdir(parents=True, exist_ok=True)
        self.cnfig.outpt.mkdir(parents=True, exist_ok=True)

    def _loadm(self) -> YOLO:
        return YOLO(self.msrce, task="detect")

    def _linez(self) -> sv.LineZone:
        return sv.LineZone(
            start=sv.Point(*self.cnfig.linea),
            end=sv.Point(*self.cnfig.lineb),
            minimum_crossing_threshold=4,
        )

    def _dayfp(self) -> Path:
        retur = datetime.now().strftime("%Y-%m-%d")
        return self.cnfig.outpt / f"{retur}.jsonl"

    @staticmethod
    def _anots() -> tuple[sv.BoxAnnotator, sv.LabelAnnotator, sv.LineZoneAnnotator]:
        return sv.BoxAnnotator(), sv.LabelAnnotator(), sv.LineZoneAnnotator()

    @staticmethod
    def _captu(sourc: str | int, scale: int) -> object:
        setup = {"resize": (scale, scale), "resize_keepratio": True}
        if Count._isnet(sourc):
            return ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, sourc, **setup)
        elif isinstance(sourc, int) or (isinstance(sourc, str) and not Path(sourc).exists()):
            return ffmpegcv.VideoCaptureCAM(sourc, **setup)
        else:
            return ffmpegcv.VideoCapture(sourc, **setup)

    @staticmethod
    def _trkid(detss: sv.Detections, index: int) -> int | None:
        trkid = detss.tracker_id
        if trkid is None or len(trkid) <= index:
            return None
        value = trkid[index]
        if value is None:
            return None
        tknbr = int(value)
        return tknbr if tknbr >= 0 else None

    @staticmethod
    def _vldid(detss: sv.Detections) -> sv.Detections:
        trkid = detss.tracker_id
        if trkid is None:
            return sv.Detections.empty()
        vmask = np.array(
            [item is not None and int(item) >= 0 for item in trkid],
            dtype=bool,
        )
        return detss[vmask]

    @staticmethod
    def _isnet(sourc: object) -> bool:
        return isinstance(sourc, str) and sourc.lower().startswith(PREFX)


def argum() -> argparse.Namespace:
    argpr = argparse.ArgumentParser(description="Line crossing counter")
    argpr.add_argument("--config", dest="confg", required=True, help="Path to config.json")
    return argpr.parse_args()


def start() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        count = Count(readc(argum().confg))
        count.runit()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        logging.exception("Runtime error: %s", error)


if __name__ == "__main__":
    start()
