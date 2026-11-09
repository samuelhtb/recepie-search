"""Dashboard evaluasi BM25: persiapan, anotasi dua penilai, metrik, grafik.

Tahap evaluasi mengikuti laporan: 10-15 resep eval, 7 variasi query,
Top-10 judgment pool, label manual 0/1/2, P@5, MRR@5, nDCG@5,
dan retrieval time. Tidak ada label otomatis dari skor BM25.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluate import VARIANTS, create_evaluation_materials, score_evaluation
from src.evaluation_dashboard import (
    annotation_progress,
    prepare_exists,
    read_judgments,
    save_annotation,
    score_is_current,
)
from src.query_processing import QUERY_LABELS
from src.ui_shared import ARTIFACTS, EVAL, PROCESSED, TRAIN

st.title("📊 Evaluasi BM25 & Grafik")
st.write(
    "Halaman ini digunakan untuk **menguji tujuh kombinasi query**, "
    "**menilai relevansi secara manual oleh dua anggota**, "
    "lalu **menghitung dan membandingkan metrik retrieval**. "
    "Hasil evaluasi tidak dibuat dari label otomatis."
)

with st.expander("📖 Metodologi evaluasi berdasarkan dokumen penelitian", expanded=False):
    st.markdown(
        """
        1. **Corpus tetap:** resep dari `train.parquet` saja; `eval.parquet` hanya menjadi sumber query.
        2. **Query uji:** pilih 10–15 resep dasar dari eval (default 12). Setiap resep
           menghasilkan Q1–Q7: judul, bahan, langkah, judul+bahan, judul+langkah,
           bahan+langkah, serta ketiganya. Default ada **84 query**.
        3. **Retrieval:** gunakan indeks BM25Okapi dan preprocessing yang sama. Simpan
           maksimal **Top-10** dokumen berskor positif dari setiap query sebagai *judgment pool*.
        4. **Ground truth:** dua penilai memberi angka **0** (tidak relevan),
           **1** (cukup relevan), atau **2** (sangat relevan). Bila berbeda,
           tentukan satu label akhir melalui diskusi.
        5. **Metrik:** hitung Precision@5, MRR@5, nDCG@5, dan waktu retrieval;
           bandingkan berdasarkan tujuh kombinasi query yang menggunakan corpus sama.
        
        **Catatan metodologis:** pemilihan 12 resep bersifat kandidat awal. Tim tetap
        perlu memastikan variasi jenis masakan dan memeriksa resep yang nyaris sama
        antar-split. Relevansi dinilai hanya untuk kandidat yang masuk pool, bukan
        seluruh dokumen dalam corpus.
        """
    )

st.caption("Panduan: **Persiapan → Penilaian manual → Hasil & Grafik**. Semua file disimpan di `data/processed/`.")

setup_tab, judge_tab, metrics_tab = st.tabs(
    ["1 · Persiapan Pengujian", "2 · Penilaian Relevansi", "3 · Hasil & Grafik"]
)


# ----------------------- TAB 1: PERSIAPAN -----------------------
with setup_tab:
    st.subheader("Siapkan pengujian Q1–Q7")
    st.write(
        "Sistem mengambil resep dasar dari **eval.parquet**, membentuk tujuh variasi "
        "query per resep, lalu mengambil hasil BM25 Top-10 untuk dinilai manusia."
    )

    ready = prepare_exists(PROCESSED)
    if ready:
        st.success("Bahan pengujian sudah tersedia. Kamu dapat langsung melanjutkan ke tab Penilaian.")
        setup_file = PROCESSED / "evaluation_setup.json"
        if setup_file.is_file():
            try:
                setup = json.loads(setup_file.read_text(encoding="utf-8"))
                c1, c2, c3 = st.columns(3)
                c1.metric("Resep dasar", setup.get("base_recipe_count", "-"))
                c2.metric("Query pengujian", setup.get("query_count", "-"))
                c3.metric("Kandidat / pool", setup.get("candidate_count", "-"))
            except (ValueError, OSError):
                st.warning("Metadata eksperimen tidak dapat dibaca, tetapi file query masih tersedia.")

    with st.form("setup_experiment"):
        n_base = st.number_input(
            "Jumlah resep dasar dari eval", min_value=10, max_value=15, value=12, step=1,
            help="Rancangan menggunakan 10–15 resep; tiap resep menghasilkan 7 query."
        )
        seed = st.number_input(
            "Random seed (agar pemilihan konsisten)", min_value=0, max_value=99999,
            value=42, step=1,
        )
        st.caption("Pool penilaian: **Top-10** untuk setiap query; tidak mengubah Top-5 pada aplikasi pencarian.")
        overwrite = st.checkbox(
            "Izinkan menimpa eksperimen dan seluruh label penilaian yang sudah tersimpan",
            disabled=not ready,
        )
        prepare_clicked = st.form_submit_button(
            "Buat / Bangun Ulang Bahan Evaluasi", type="primary", use_container_width=True
        )

    if prepare_clicked:
        if ready and not overwrite:
            st.error("Data evaluasi lama sudah ada. Centang izin penimpaan jika benar-benar ingin mengulang.")
        elif not TRAIN.is_file() or not EVAL.is_file():
            st.error("File train.parquet dan eval.parquet harus berada di folder data/raw/.")
        elif not (ARTIFACTS / "manifest.json").is_file():
            st.error("Indeks BM25 belum dibuat. Jalankan `python -m src.indexing` lebih dahulu.")
        else:
            try:
                with st.spinner("Membangun query dan melakukan retrieval untuk setiap kombinasi. Ini dapat memakan waktu..."):
                    metadata = create_evaluation_materials(
                        eval_path=EVAL,
                        train_path=TRAIN,
                        artifacts_dir=ARTIFACTS,
                        output_dir=PROCESSED,
                        n_base=int(n_base),
                        pool_k=10,
                        seed=int(seed),
                    )
                # Setelah eksperimen dibangun ulang, metrik lama harus dihapus:
                # nilainya tidak lagi mewakili query dan pool terbaru.
                for file in ("metrics_by_query.csv", "metrics_by_combination.csv", "evaluation_summary.json"):
                    (PROCESSED / file).unlink(missing_ok=True)
                st.session_state["evaluation_notice"] = (
                    f"Bahan evaluasi berhasil dibuat: {metadata['query_count']} query dan "
                    f"{metadata['candidate_count']} kandidat penilaian."
                )
                st.rerun()
            except (ValueError, FileNotFoundError, RuntimeError, OSError, ModuleNotFoundError) as exc:
                st.error(f"Gagal menyiapkan evaluasi: {exc}")

    with st.expander("Lihat tujuh bentuk query yang diuji"):
        st.dataframe(
            pd.DataFrame(
                [{"Kode": code, "Bentuk query": QUERY_LABELS[code]} for code in VARIANTS]
            ),
            hide_index=True, use_container_width=True,
        )
    st.info(
        "**Mengapa menggunakan query dari eval?** Agar resep sumber query tidak ikut "
        "diindeks sebagai dokumen latih. Program juga mengecualikan resep eval "
        "yang identik dengan resep train saat menyiapkan query uji."
    )


# ----------------------- READ AND PROGRESS -----------------------
queries_path = PROCESSED / "queries.csv"
pool_path = PROCESSED / "retrieval_pool.csv"
judgments_path = PROCESSED / "relevance_judgments.csv"
materials_ready = prepare_exists(PROCESSED)
queries: pd.DataFrame | None = None
pool: pd.DataFrame | None = None
judgments: pd.DataFrame | None = None
progress: dict | None = None
load_error = None

if materials_ready:
    try:
        queries = pd.read_csv(queries_path, dtype=str, keep_default_na=False)
        pool = pd.read_csv(pool_path, dtype=str, keep_default_na=False)
        judgments = read_judgments(judgments_path)
        progress = annotation_progress(judgments)
    except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
        load_error = str(exc)


# ----------------------- TAB 2: ANOTASI -----------------------
with judge_tab:
    st.subheader("Anotasi manual oleh dua penilai")
    st.markdown(
        """
        Setiap kandidat **wajib dinilai oleh dua anggota kelompok**. Penilai sebaiknya
        mengisi nilainya masing-masing secara independen sebelum berdiskusi.
        Pilih peran penilai, pilih query, periksa judul/bahan/langkah kandidat,
        isi nilai melalui tabel, lalu klik **Simpan Penilaian**. Nilai tersimpan
        dalam file CSV, sehingga tetap ada setelah aplikasi ditutup.
        """
    )
    with st.expander("Pedoman pemberian nilai 0–2", expanded=True):
        st.markdown(
            """
            | Nilai | Kategori | Pedoman |
            |---|---|---|
            | **0** | Tidak relevan | Resep tidak sesuai kebutuhan dalam query. |
            | **1** | Cukup relevan | Resep memiliki sebagian karakteristik yang cocok. |
            | **2** | Sangat relevan | Resep sangat sesuai dengan kebutuhan informasi. |

            Contoh: jika query *Ayam Goreng Bawang Putih*, masakan dengan jenis
            hidangan dan bumbu yang sangat sesuai dapat dinilai 2; resep yang
            hanya cocok sebagian dinilai 1; resep tak sesuai dinilai 0.
            **Skor BM25 tidak menentukan label penilai.** Nilai akhir hanya
            perlu diisi ketika dua penilai berbeda pendapat.
            """
        )
    if load_error:
        st.error(f"Gagal membaca data evaluasi: {load_error}")
    elif not materials_ready or judgments is None or queries is None or progress is None:
        st.info("Belum ada query evaluasi. Buat terlebih dahulu pada tab Persiapan Pengujian.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kandidat penilaian", progress["total"])
        c2.metric("Penilai 1 terisi", f"{progress['rater_1_done']} / {progress['total']}")
        c3.metric("Penilai 2 terisi", f"{progress['rater_2_done']} / {progress['total']}")
        c4.metric("Konflik belum selesai", progress["conflicts_unresolved"])
        st.progress(progress["ready_count"] / progress["total"] if progress["total"] else 0)
        st.caption(
            f"Siap dihitung: **{progress['ready_count']} / {progress['total']}** kandidat. "
            "Keduanya harus memberi label dan perbedaan harus disepakati."
        )
        if progress["complete"]:
            st.success("Seluruh kandidat selesai dinilai. Silakan ke tab Hasil & Grafik.")
        elif progress["invalid_final"]:
            st.error("Ada keputusan akhir yang tidak sesuai dengan nilai dua penilai yang sepakat.")

        left, right = st.columns([1, 2])
        with left:
            role_label = st.radio(
                "Mode penilaian",
                ["Penilai 1", "Penilai 2", "Diskusi / nilai akhir"],
                help="Penilai 1 dan 2 memiliki kolom terpisah. Nilai akhir hanya untuk pasangan yang berbeda."
            )
        role = {
            "Penilai 1": "rater_1",
            "Penilai 2": "rater_2",
            "Diskusi / nilai akhir": "final_relevance",
        }[role_label]

        qlabels = {
            row.query_id: f"{row.query_code} · {row.base_recipe_id} · {row.query_type}"
            for row in queries.itertuples(index=False)
        }
        with right:
            query_id = st.selectbox(
                "Query yang sedang dinilai",
                options=list(qlabels),
                format_func=lambda value: qlabels[value],
            )
        query = queries.loc[queries["query_id"] == query_id].iloc[0]
        with st.expander("Lihat informasi query uji", expanded=True):
            if query["title"]:
                st.markdown(f"**Judul:** {query['title']}")
            if query["ingredients"]:
                st.markdown("**Bahan:**")
                st.text(query["ingredients"])
            if query["steps"]:
                st.markdown("**Langkah:**")
                st.text(query["steps"])

        selected = judgments.loc[judgments["query_id"] == query_id].copy()
        selected["rank"] = pd.to_numeric(selected["rank"], errors="coerce")
        selected = selected.sort_values("rank")

        if role == "final_relevance":
            selected = selected.loc[
                (selected["rater_1"] != "")
                & (selected["rater_2"] != "")
                & (selected["rater_1"] != selected["rater_2"])
            ]
            st.caption(
                "Di sini hanya tampil dokumen yang kedua penilainya tidak sepakat. "
                "Diskusikan isinya lalu tetapkan nilai akhir 0, 1, atau 2."
            )
        else:
            st.caption(
                "Kamu hanya melihat/mengedit kolom penilai yang dipilih, "
                "untuk membantu menjaga penilaian awal tetap independen."
            )

        if selected.empty:
            st.info("Tidak ada kandidat yang perlu dinilai untuk mode/query ini.")
        else:
            columns = ["doc_id", "rank", "recipe_title", "bm25_score", role]
            edited_input = selected[columns].copy()
            editor_key = (
                f"evaluation_editor_{role}_{query_id}_"
                f"{st.session_state.get('annotation_editor_revision', 0)}"
            )
            with st.form("annotation_form"):
                edited = st.data_editor(
                    edited_input,
                    key=editor_key,
                    hide_index=True,
                    use_container_width=True,
                    num_rows="fixed",
                    disabled=["doc_id", "rank", "recipe_title", "bm25_score"],
                    column_config={
                        "doc_id": st.column_config.TextColumn("ID Dokumen", width="medium"),
                        "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                        "recipe_title": st.column_config.TextColumn("Judul resep", width="large"),
                        "bm25_score": st.column_config.TextColumn("Skor BM25", width="small"),
                        role: st.column_config.SelectboxColumn(
                            "Nilai relevansi (0 / 1 / 2)",
                            options=["", "0", "1", "2"],
                            required=False,
                            width="medium",
                        ),
                    },
                )
                save_clicked = st.form_submit_button("💾 Simpan Penilaian", type="primary")
            if save_clicked:
                try:
                    changes = save_annotation(judgments_path, query_id, role, edited)
                    st.session_state["evaluation_notice"] = (
                        f"Berhasil menyimpan {changes} perubahan untuk {role_label}."
                    )
                    st.session_state["annotation_editor_revision"] = (
                        st.session_state.get("annotation_editor_revision", 0) + 1
                    )
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(f"Penilaian gagal disimpan: {exc}")

            st.markdown("**Baca resep lengkap sebelum memberikan label:**")
            for row in selected.itertuples(index=False):
                with st.expander(f"Rank {row.rank} — {row.recipe_title} ({row.doc_id})"):
                    st.markdown("**Bahan:**")
                    st.text(row.recipe_ingredients or "(tidak tersedia)")
                    st.markdown("**Langkah memasak:**")
                    st.text(row.recipe_steps or "(tidak tersedia)")
                    st.caption(f"Skor BM25 {row.bm25_score} — bukan nilai relevansi.")

        st.download_button(
            "⬇️ Unduh tabel anotasi CSV (untuk cadangan/kolaborasi)",
            data=judgments_path.read_bytes(),
            file_name="relevance_judgments.csv",
            mime="text/csv",
        )
        st.caption("Alternatif: CSV dapat diedit di luar aplikasi lalu diletakkan kembali di `data/processed/`.")


# ----------------------- TAB 3: SKOR & VISUALISASI -----------------------
with metrics_tab:
    st.subheader("Hasil evaluasi dan interpretasi")
    st.write(
        "Perhitungan hanya bisa dilakukan **setelah seluruh kandidat Top-10 dinilai** "
        "oleh dua orang dan setiap perbedaan diselesaikan. Angka dan grafik tidak "
        "akan dibuat dari label buatan atau nilai BM25 sebagai pengganti relevansi."
    )

    with st.expander("📘 Penjelasan metrik: rumus dan cara membaca", expanded=True):
        st.markdown(
            """
            **Precision@5 (P@5)** — proporsi hasil relevan di posisi 1–5.
            Nilai **1 atau 2** dihitung relevan, nilai **0** tidak relevan.
            Misalnya 4 dari lima hasil relevan → P@5 = **0,80**.
            Jika hasil kurang dari lima, pembaginya tetap 5 sesuai laporan.

            **MRR@5 (Mean Reciprocal Rank)** — mengukur seberapa cepat
            dokumen relevan *pertama* ditemukan dalam Top-5.
            Peringkat 1 → 1,00; peringkat 2 → 0,50; tidak ada relevan di
            Top-5 → 0. Hasil akhir adalah rata-rata seluruh query.

            **nDCG@5** — mengukur *kualitas urutan* dengan mempertimbangkan
            nilai bertingkat 0, 1, dan 2. Hasil yang sangat relevan (2) di
            ranking atas mendapat kontribusi lebih besar. Nilai 1 mendekati
            urutan ideal berdasarkan kandidat yang sudah dinilai.

            **Retrieval time (ms)** — waktu dalam milidetik untuk validasi,
            preprocessing query, perhitungan skor dan ranking BM25 (pada tahap
            pembentukan pool). Semakin kecil semakin cepat, tetapi angka
            dipengaruhi perangkat dan panjang query.
            """
        )
        st.latex(r"P@5 = \frac{\sum_{i=1}^{5}\mathbb{1}(rel_i\geq 1)}{5}")
        st.latex(r"MRR@5 = \frac{1}{Q}\sum_{q=1}^{Q}\frac{1}{rank_q}\quad (0\;\text{bila tidak ada relevan})")
        st.latex(r"DCG@5 = \sum_{i=1}^{5}\frac{2^{rel_i}-1}{\log_2(i+1)},\qquad nDCG@5 = \frac{DCG@5}{IDCG@5}")
        st.caption(
            "IDCG@5 dihitung dari relevansi terbaik dalam *judgment pool* yang dinilai "
            "(umumnya Top-10), bukan dari seluruh corpus. Jika IDCG = 0, nDCG = 0."
        )

    if load_error:
        st.error(f"Data evaluasi belum valid: {load_error}")
    elif not materials_ready or progress is None:
        st.info("Mulai dari tab Persiapan Pengujian untuk membuat query dan kandidat.")
    else:
        if not progress["complete"]:
            st.warning(
                f"Penilaian belum selesai: **{progress['ready_count']} dari {progress['total']}** kandidat siap. "
                "Lengkapi tab Penilaian Relevansi terlebih dahulu."
            )
        score_clicked = st.button(
            "🧮 Hitung / Perbarui Metrik Evaluasi",
            type="primary",
            disabled=not progress["complete"],
        )
        if score_clicked:
            try:
                with st.spinner("Menghitung metrik dari seluruh penilaian manual..."):
                    summary = score_evaluation(
                        queries_csv=queries_path,
                        pool_csv=pool_path,
                        judgments_csv=judgments_path,
                        out_dir=PROCESSED,
                    )
                st.session_state["evaluation_notice"] = (
                    f"Perhitungan selesai untuk {summary['queries_evaluated']} query."
                )
                st.rerun()
            except (ValueError, OSError, KeyError) as exc:
                st.error(f"Metrik gagal dihitung: {exc}")

        if progress["complete"] and not score_is_current(PROCESSED):
            st.info("Belum ada hasil metrik terbaru. Klik **Hitung / Perbarui Metrik Evaluasi**.")
        elif progress["complete"] and score_is_current(PROCESSED):
            summary = json.loads((PROCESSED / "evaluation_summary.json").read_text(encoding="utf-8"))
            by_combo = pd.read_csv(PROCESSED / "metrics_by_combination.csv")
            by_query = pd.read_csv(PROCESSED / "metrics_by_query.csv")
            order = list(VARIANTS)
            by_combo["query_code"] = pd.Categorical(by_combo["query_code"], categories=order, ordered=True)
            by_combo = by_combo.sort_values("query_code")
            by_query["query_code"] = pd.Categorical(by_query["query_code"], categories=order, ordered=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rata-rata Precision@5", f"{summary['mean_precision_at_5']:.3f}")
            c2.metric("MRR@5", f"{summary['mrr_at_5']:.3f}")
            c3.metric("Rata-rata nDCG@5", f"{summary['mean_ndcg_at_5']:.3f}")
            c4.metric("Waktu rata-rata", f"{summary['mean_retrieval_time_ms']:.1f} ms")
            st.caption(
                f"Hasil berdasarkan **{summary['queries_evaluated']} query**, "
                f"{summary['judged_candidates']} dokumen-query yang dinilai. "
                "Metrik relevansi berkisar antara 0 (rendah) dan 1 (tinggi)."
            )

            # GRAFIK 1: bar grouped untuk 3 metrik yang skalanya sama [0,1].
            st.markdown("#### Grafik 1 · Perbandingan kualitas retrieval Q1–Q7")
            graph_data = by_combo.melt(
                id_vars="query_code",
                value_vars=["mean_precision_at_5", "mrr_at_5", "mean_ndcg_at_5"],
                var_name="Metrik", value_name="Skor",
            )
            graph_data["Metrik"] = graph_data["Metrik"].replace({
                "mean_precision_at_5": "Precision@5",
                "mrr_at_5": "MRR@5",
                "mean_ndcg_at_5": "nDCG@5",
            })
            fig_quality = px.bar(
                graph_data, x="query_code", y="Skor", color="Metrik",
                barmode="group", text_auto=".2f",
                category_orders={"query_code": order},
            )
            fig_quality.update_layout(
                yaxis=dict(range=[0, 1.10]), xaxis_title="Kombinasi query",
                legend_title="Metrik", height=460,
            )
            st.plotly_chart(fig_quality, use_container_width=True)
            st.caption(
                "Semakin tinggi batang, semakin baik metrik tersebut. Bandingkan satu "
                "metrik antar-Q1–Q7; ketiganya mengukur aspek berbeda, sehingga "
                "tidak boleh ditafsirkan sebagai satu nilai akurasi."
            )

            # GRAFIK 2: latency terpisah supaya satuannya tidak dicampur dengan 0-1.
            st.markdown("#### Grafik 2 · Perbandingan waktu retrieval")
            time_data = by_combo.melt(
                id_vars="query_code",
                value_vars=["mean_retrieval_time_ms", "median_retrieval_time_ms"],
                var_name="Statistik", value_name="Waktu (ms)",
            )
            time_data["Statistik"] = time_data["Statistik"].replace({
                "mean_retrieval_time_ms": "Rata-rata", "median_retrieval_time_ms": "Median"
            })
            fig_time = px.bar(
                time_data, x="query_code", y="Waktu (ms)", color="Statistik",
                barmode="group", text_auto=".1f", category_orders={"query_code": order},
            )
            fig_time.update_layout(xaxis_title="Kombinasi query", height=430)
            st.plotly_chart(fig_time, use_container_width=True)
            st.caption(
                "Waktu lebih kecil berarti lebih cepat pada perangkat yang diuji. "
                "Perbedaan waktu dapat dipengaruhi banyaknya token dalam query, "
                "beban laptop, serta variasi kondisi eksekusi."
            )

            # GRAFIK 3: variasi per resep dasar; tidak hanya rata-rata agregat.
            st.markdown("#### Grafik 3 · Heatmap nDCG@5 untuk setiap resep dasar")
            matrix = by_query.pivot(
                index="base_recipe_id", columns="query_code", values="ndcg_at_5"
            ).reindex(columns=order)
            fig_heat = px.imshow(
                matrix, zmin=0, zmax=1, aspect="auto", text_auto=".2f",
                labels={"x": "Kombinasi query", "y": "Resep dasar", "color": "nDCG@5"},
                color_continuous_scale="Blues",
            )
            fig_heat.update_layout(height=max(400, 34 * len(matrix) + 145))
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                "Setiap baris merepresentasikan resep sumber query yang sama untuk "
                "seluruh Q1–Q7. Warna lebih pekat menandakan urutan hasil lebih "
                "mendekati ideal berdasarkan kandidat yang diberi label. "
                "Grafik ini menunjukkan bahwa hasil dapat berbeda antar-topik resep."
            )

            # Penjelasan hasil otomatis yang langsung berbasis angka nyata (bukan klaim kausal).
            max_ndcg = by_combo["mean_ndcg_at_5"].max()
            top_codes = by_combo.loc[
                by_combo["mean_ndcg_at_5"].eq(max_ndcg), "query_code"
            ].astype(str).tolist()
            min_time = by_combo["mean_retrieval_time_ms"].min()
            fastest = by_combo.loc[
                by_combo["mean_retrieval_time_ms"].eq(min_time), "query_code"
            ].astype(str).tolist()
            st.markdown("#### Interpretasi hasil eksperimen")
            st.write(
                f"Berdasarkan judgment pool yang telah dinilai, nilai **nDCG@5 rata-rata "
                f"tertinggi ({max_ndcg:.3f})** diperoleh oleh "
                f"**{', '.join(f'{code} ({QUERY_LABELS[code]})' for code in top_codes)}**. "
                "Ini menunjukkan urutan hasil yang paling mendekati urutan ideal "
                "pada kumpulan query uji yang tersedia, **bukan bukti universal bahwa "
                "kombinasi tersebut selalu terbaik**."
            )
            st.write(
                f"**Waktu retrieval rata-rata terendah ({min_time:.1f} ms)** "
                f"tercatat pada **{', '.join(f'{code} ({QUERY_LABELS[code]})' for code in fastest)}**. "
                "Kecepatan tidak selalu sejalan dengan relevansi. Perbandingan akhir "
                "perlu mempertimbangkan kedua aspek."
            )
            st.warning(
                "**Batas interpretasi:** angka ini berasal dari 10–15 resep dasar dan "
                "kandidat maksimal Top-10 per query, bukan penilaian terhadap semua "
                "resep relevan dalam corpus. Skor BM25 bukan label relevansi; "
                "hasil tidak boleh diklaim sebagai evaluasi lengkap seluruh koleksi."
            )

            with st.expander("Lihat tabel nilai semua kombinasi", expanded=False):
                rename = {
                    "query_code": "Kode",
                    "query_type": "Jenis Query",
                    "total_queries": "Jumlah Query",
                    "mean_precision_at_5": "P@5",
                    "mrr_at_5": "MRR@5",
                    "mean_ndcg_at_5": "nDCG@5",
                    "mean_retrieval_time_ms": "Waktu rerata (ms)",
                    "median_retrieval_time_ms": "Waktu median (ms)",
                }
                st.dataframe(by_combo.rename(columns=rename), hide_index=True, use_container_width=True)
                st.caption("Nilai rata-rata setiap kombinasi dihitung dari query yang memakai resep dasar sama.")

            view_code = st.selectbox(
                "Lihat detail hasil per query", options=order,
                format_func=lambda code: f"{code} — {QUERY_LABELS[code]}",
                key="query_score_filter",
            )
            st.dataframe(
                by_query.loc[by_query["query_code"] == view_code],
                hide_index=True, use_container_width=True,
            )
            st.download_button(
                "⬇️ Unduh ringkasan per kombinasi (CSV)",
                (PROCESSED / "metrics_by_combination.csv").read_bytes(),
                file_name="metrics_by_combination.csv", mime="text/csv",
            )
            st.download_button(
                "⬇️ Unduh hasil per query (CSV)",
                (PROCESSED / "metrics_by_query.csv").read_bytes(),
                file_name="metrics_by_query.csv", mime="text/csv",
            )

# Notifikasi setelah rerun diletakkan di akhir agar tampil satu kali.
notice = st.session_state.pop("evaluation_notice", None)
if notice:
    st.toast(notice, icon="✅")
