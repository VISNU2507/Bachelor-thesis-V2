import sys
import pandas as pd
project_root = "C:/Users/Visnu/DIAMONDS"  

if project_root not in sys.path:
    sys.path.append(project_root)
from diamonds import data as dt
import numpy as np
import matplotlib.pyplot as plt
from diamonds_definitions import pt

class GoldenBreathAnalyzer:
    def __init__(self, patient, session_type=dt.SeatedSession,
                 interval_size: float = 10, step_len: float = 5):
        self.patient      = patient
        self.session      = patient[session_type]
        self.breaths      = self.session[dt.Breaths]
        self.exercise     = self.session[dt.Exercise]
        self.interval_size = interval_size
        self.step_len      = step_len

 
        self.in_times, self.ex_times = self._get_golden_breath_events()
        self.start_time = float(np.min(self.exercise.timestamp))
        self.end_time   = float(np.max(self.exercise.timestamp)) + 30

        print(f"Exercise interval: {self.start_time:.1f}s to {self.end_time:.1f}s")

    def _get_golden_breath_events(self):
        in_t = np.array([t for t,l in zip(self.breaths.timestamp, self.breaths.ann) if l=='(In'])
        ex_t = np.array([t for t,l in zip(self.breaths.timestamp, self.breaths.ann) if l=='(Ex'])
        return in_t, ex_t

    def _annotate_exercises(self, ax):
        for tstamp, label in zip(self.exercise.timestamp, self.exercise.ann):
            ax.axvline(tstamp, color='magenta', linestyle='--', alpha=0.5)
            ax.text(tstamp, ax.get_ylim()[1]*0.95, label,
                    rotation=90, va='top', color='magenta')

    def _compute_bpm(self, t_start, t_end):
        valid_in = self.in_times[(self.in_times >= t_start) & (self.in_times <= t_end)]
        if len(valid_in) < 1:
            return np.nan
        return 60.0 / np.mean(np.diff(valid_in))


    def segment_bpm(self):
        seg_starts = np.arange(
            self.start_time,
            self.end_time - self.interval_size + 1e-9,
            self.step_len
        )
        bpm_vals = np.full(seg_starts.size, np.nan)

        for k, t0 in enumerate(seg_starts):
            t1 = t0 + self.interval_size
            these = self.in_times[(self.in_times >= t0) & (self.in_times < t1)]
            if these.size >= 2:
                bpm_vals[k] = 60.0 / np.mean(np.diff(these))

        
                # linear fill — husk at interpolate only missing values
        good = ~np.isnan(bpm_vals)
        if np.any(good):
            interp_vals = np.interp(seg_starts, seg_starts[good], bpm_vals[good])
            bpm_vals[~good] = interp_vals[~good]

        return seg_starts, bpm_vals

        return seg_starts, bpm_vals


    def segment_true_insp_exp_durations(self):
        seg_starts = np.arange(
            self.start_time,
            self.end_time - self.interval_size + 1e-9,
            self.step_len
        )
        insp = np.full(seg_starts.size, np.nan)
        exp  = np.full(seg_starts.size, np.nan)

        for k, t0 in enumerate(seg_starts):
            t1 = t0 + self.interval_size

            # true inspiration (Ex→In)
            i_durs = []
            for ex in self.ex_times[(self.ex_times >= t0) & (self.ex_times < t1)]:
                nxt = self.in_times[self.in_times > ex]
                if nxt.size and nxt[0] <= t1:
                    i_durs.append(nxt[0] - ex)

            # true expiration (In→Ex)
            e_durs = []
            for ino in self.in_times[(self.in_times >= t0) & (self.in_times < t1)]:
                nxt = self.ex_times[self.ex_times > ino]
                if nxt.size and nxt[0] <= t1:
                    e_durs.append(nxt[0] - ino)

            if i_durs: insp[k] = np.mean(i_durs)
            if e_durs: exp[k]  = np.mean(e_durs)

        for arr in (insp, exp):
            if np.any(np.isnan(arr)):
                good = ~np.isnan(arr)
                arr[~good] = np.interp(seg_starts[~good],
                                       seg_starts[good],
                                       arr[good])
        return seg_starts, insp, exp


    def plot_bpm(self, seg_times, bpm_vals):
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        ax.plot(seg_times, bpm_vals, '-o', label='Golden BPM (Inspiration)')
        ax.set_title("Golden Standard: BPM in Exercise Intervals")
        ax.set_xlabel("Segment Start Time (s)")
        ax.set_ylabel("Breaths Per Minute")
        ax.grid(True)
        ax.legend()
        self._annotate_exercises(ax)
        plt.show()

    def plot_true_insp_exp_durations(self, seg_times, true_insp, true_exp):
        plt.figure(figsize=(10, 4))
        plt.plot(seg_times, true_insp, '-o', label="True Inspiration (Ex→In)")
        plt.plot(seg_times, true_exp, '-o', label="True Expiration (In→Ex)")
        plt.title("True Inspiration & Expiration Durations (Gold Standard)")
        plt.xlabel("Segment Start Time (s)")
        plt.ylabel("Duration (s)")
        plt.grid(True)
        plt.legend()
        self._annotate_exercises(plt.gca())
        plt.show()

    def plot_ie_ratio(self, seg_times, ie_ratios):
        plt.figure(figsize=(8, 4))
        plt.plot(seg_times, ie_ratios, '-o', color='orange', label="I:E Ratio")
        plt.title("Inspiration to Expiration Ratio (I:E)")
        plt.xlabel("Segment Start Time (s)")
        plt.ylabel("Ratio")
        plt.grid(True)
        plt.legend()
        self._annotate_exercises(plt.gca())
        plt.show()
    
    def define_complete_breath_cycles(self):

        breath_starts = []
        breath_ends = []
        insp_durations = []
        exp_durations = []

        for i in range(len(self.ex_times) - 1):
            start_ex = self.ex_times[i]
            next_in = self.in_times[self.in_times > start_ex]
            if next_in.size == 0:
                continue
            in_time = next_in[0]

            next_ex = self.ex_times[self.ex_times > in_time]
            if next_ex.size == 0:
                continue
            end_ex = next_ex[0]

            breath_starts.append(start_ex)
            breath_ends.append(end_ex)
            insp_durations.append(in_time - start_ex)
            exp_durations.append(end_ex - in_time)

        return (
            np.array(breath_starts),
            np.array(breath_ends),
            np.array(insp_durations),
            np.array(exp_durations)
        )

    def print_total_breath_count(self):
        total = len(self.breaths.ann)
        total_in = len(self.in_times)
        total_ex = len(self.ex_times)
        est_cycles = min(total_in, total_ex)
        print(f"Total breath events: {total}")
        print(f"Inspirations: {total_in}, Expirations: {total_ex}")
        print(f"Estimated breath cycles (In↔Ex pairs): {est_cycles}")
    
    def print_breath_counts_per_interval(self):
        seg_times = np.arange(self.start_time, self.end_time, self.interval_size)
        print(f"\nBreath event counts per {self.interval_size}-second interval:")
        for t0 in seg_times:
            t1 = t0 + self.interval_size
            mask = (np.array(self.breaths.timestamp) >= t0) & (np.array(self.breaths.timestamp) < t1)
            print(f"[{t0:.0f}-{t1:.0f}s]: {np.sum(mask)} events")

    def run_analysis(self):
        print("\n=== Running Golden Breath Analysis ===")

 
        bpm_times, bpm_vals = self.segment_bpm()
        self.plot_bpm(bpm_times, bpm_vals)

    
        seg_times, true_insp, true_exp = self.segment_true_insp_exp_durations()
        self.plot_true_insp_exp_durations(seg_times, true_insp, true_exp)

       
        ie_ratios = true_insp / true_exp
        self.plot_ie_ratio(seg_times, ie_ratios)

  
        print("\n=== Segment-wise Duration Summary ===")
        for t0, insp, exp, ratio in zip(seg_times, true_insp, true_exp, ie_ratios):
            t1 = t0 + self.interval_size
            print(f"[{t0:.0f}-{t1:.0f}s] Insp={insp:.2f}s, Exp={exp:.2f}s, I:E={ratio:.2f}")

     
        print("\n=== BPM Summary ===")
        for t0, bpm in zip(bpm_times, bpm_vals):
            t1 = t0 + self.interval_size
            print(f"[{t0:.0f}-{t1:.0f}s]: BPM={bpm:.2f}")

   
        self.print_total_breath_count()
        self.print_breath_counts_per_interval()

      
        starts, ends, insp_durs, exp_durs = self.define_complete_breath_cycles()
        print(f"\n✅ Total valid breath cycles (Ex→In→Ex): {len(starts)}")

    def build_golden_table(self) -> pd.DataFrame:
       
        seg_starts, bpm_vals = self.segment_bpm()
        _, insp_vals, exp_vals = self.segment_true_insp_exp_durations()

        
        starts, ends, insp_durs, exp_durs = self.define_complete_breath_cycles()
        cycle_counts = []
        avg_totals   = []
        for t0 in seg_starts:
            t1 = t0 + self.interval_size
            mask = (starts >= t0) & (ends < t1)
            cycle_counts.append(mask.sum())
            tot_dur = (insp_durs + exp_durs)[mask]
            avg_totals.append(np.mean(tot_dur) if len(tot_dur)>0 else np.nan)

        rows = []
        for ws, we, bpm, cnt, avg, ti, te, ie in zip(
            seg_starts,
            seg_starts + self.interval_size,
            bpm_vals,
            cycle_counts,
            avg_totals,
            insp_vals,
            exp_vals,
            insp_vals/exp_vals
        ):
            rows.append({
                "subject_id":             self.patient.id,
                "window_start":           ws,
                "window_end":             we,
                "gold_BPM":               bpm,
                "gold_n_breaths":         cnt,
                "gold_AvgTotalBreathDuration": avg,
                "gold_TrueInspDuration":  ti,
                "gold_TrueExpDuration":   te,
                "gold_IEratio":           ie
            })

        df = pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)

        
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].interpolate(method="linear", limit_direction="both", axis=0)

       
        df["gold_n_breaths"] = df["gold_n_breaths"].fillna(0).astype(int)

        return df
    def define_complete_breath_cycles(self):
            starts, ends, i_durs, e_durs = [], [], [], []
            for i in range(len(self.ex_times)-1):
                ex0 = self.ex_times[i]
                nxt_in = self.in_times[self.in_times > ex0]
                if not len(nxt_in): continue
                in_t = nxt_in[0]

                nxt_ex = self.ex_times[self.ex_times > in_t]
                if not len(nxt_ex): continue
                ex1 = nxt_ex[0]

                starts.append(ex0)
                ends.append(ex1)
                i_durs.append(in_t-ex0)
                e_durs.append(ex1-in_t)

            return (np.array(starts), np.array(ends),
                    np.array(i_durs), np.array(e_durs))