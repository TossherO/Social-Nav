"""On-screen video alignment marker; never touches a sensor or chassis."""
import argparse
import json
from pathlib import Path
import signal
import time


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',required=True)
    parser.add_argument('--ready',required=True)
    parser.add_argument('--stop-file',required=True)
    parser.add_argument('--period-ms',type=int,default=500)
    parser.add_argument('--max-seconds',type=float,default=90.)
    args=parser.parse_args(argv)
    if not 250<=args.period_ms<=1000 or not 2<=args.max_seconds<=120:
        parser.error('Invalid marker period or maximum duration')

    import tkinter as tk
    events=Path(args.events);ready=Path(args.ready);stop=Path(args.stop_file)
    root=tk.Tk()
    root.title('SOCIAL-NAV VIDEO SYNC')
    root.geometry('460x230+40+40')
    root.attributes('-topmost',True)
    label=tk.Label(root,font=('DejaVu Sans',40,'bold'),anchor='center')
    label.pack(fill='both',expand=True)
    began=time.monotonic()
    should_stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:should_stop.__setitem__(0,True))
    root.protocol('WM_DELETE_WINDOW',lambda:should_stop.__setitem__(0,True))
    with events.open('x',buffering=1) as stream:
        sequence=[0]

        def tick():
            if should_stop[0] or stop.exists() or time.monotonic()-began>=args.max_seconds:
                root.destroy();return
            seq=sequence[0];sequence[0]+=1
            white=(seq%2)==0
            before=time.monotonic()
            label.configure(text='SYNC\n%06d'%seq,bg='white' if white else 'black',fg='black' if white else 'white')
            root.update_idletasks()
            after=time.monotonic()
            row=dict(marker_id=seq,white=white,draw_request_mono=before,draw_queued_mono=after,
                wall_s=time.time(),clock_basis='host_monotonic_gui_requests_not_screen_emission')
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            temporary=ready.with_name(ready.name+'.tmp')
            temporary.write_text(json.dumps(dict(marker_id=seq,draw_queued_mono=after)))
            temporary.replace(ready)
            root.after(args.period_ms,tick)

        root.after(100,tick)
        try:root.mainloop()
        finally:
            if ready.exists():ready.unlink()
    return 0


if __name__=='__main__':raise SystemExit(main())
