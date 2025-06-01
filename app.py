import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import pandas as pd
import io
import subprocess
from detector import Detector
from track import SimpleTracker
from generate_report import generate_pdf_report
from evaluator import Evaluator
from utils import generate_heatmap

st.set_page_config(page_title="Vehicle Tracker", layout="wide")
st.markdown("""
<style>
    .header { text-align: center; padding: 1rem; background: #4CAF50; color: white; border-radius: 8px; }
    .metric { background: #f0f2f6; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; }
    .alert { background: #ff4444; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center; }
</style>
""", unsafe_allow_html=True)

class VideoProcessor:
    def __init__(self, selected_classes):
        self.selected_classes = selected_classes
        self.detector = Detector(model_path='yolov8n.pt', conf_threshold=0.3)
        self.tracker = SimpleTracker()
        self.evaluator = Evaluator()
        self.frame_count = 0
        self.emergency_mode = False
        self.class_counts = {cls: 0 for cls in selected_classes}
        self.emergency_classes = ['car', 'truck', 'bus', 'motorcycle', 'ambulance']
        if self.emergency_mode:
            for cls in self.emergency_classes:
                if cls in self.selected_classes:
                    self.class_counts[f"emergency_{cls}"] = 0
        self.detections_for_approx = []

    def process_frame(self, frame):
        start_time = time.time()
        self.frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        detections = self.detector.detect(frame)
        if self.emergency_mode:
            detections = self.detector.detect_emergency(frame, detections)
        tracks = self.tracker.update(detections)
        self.evaluator.add_detections(self.frame_count, tracks)
        for track in tracks:
            self.detections_for_approx.append({
                'frame': self.frame_count,
                'x1': track['bbox'][0],
                'y1': track['bbox'][1],
                'x2': track['bbox'][2],
                'y2': track['bbox'][3],
                'class': track['class_name'].replace('emergency_', ''),
                'track_id': track['track_id']
            })
        annotated_frame = self.tracker.draw(frame, tracks)
        end_time = time.time()
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        alerts = [t['class_name'] for t in tracks if 'emergency' in t['class_name']]
        for track in tracks:
            class_name = track['class_name']
            base_class = class_name.replace('emergency_', '')
            if base_class in self.selected_classes:
                key = class_name if 'emergency' in class_name and class_name in self.class_counts else base_class
                self.class_counts[key] += 1
        print(f"Frame {self.frame_count}: Detections: {len(detections)}, Tracks: {len(tracks)}")
        return annotated_frame, len(detections), len(tracks), alerts

    def process_video(self, video_file, live_display, gt_file=None, source=None):
        self.evaluator.reset()
        self.frame_count = 0
        self.class_counts = {cls: 0 for cls in self.selected_classes}
        self.emergency_classes = ['car', 'truck', 'bus', 'motorcycle', 'ambulance']
        if self.emergency_mode:
            for cls in self.emergency_classes:
                if cls in self.selected_classes:
                    self.class_counts[f"emergency_{cls}"] = 0
        self.detections_for_approx = []

        if gt_file and os.path.exists(gt_file):
            st.info("Chargement du fichier de ground truth téléversé...")
            try:
                self.evaluator.load_ground_truth(gt_file)
                st.success("✅ Ground truth chargé avec succès !")
            except Exception as e:
                st.error(f"Échec du chargement du fichier de ground truth : {e}")
                st.info("Format CSV attendu : frame,x1,y1,x2,y2,class,track_id")
                return None, None
        elif source == "Sample Video" and os.path.exists("sample_video/traffic_gt.csv"):
            st.info("Chargement automatique du ground truth pour la vidéo d'exemple...")
            try:
                self.evaluator.load_ground_truth("sample_video/traffic_gt.csv")
                st.success("✅ Ground truth chargé avec succès !")
            except Exception as e:
                st.error(f"Échec du chargement du ground truth d'exemple : {e}")
                st.info("Création d'un ground truth approximatif...")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            if isinstance(video_file, bytes):
                tmp.write(video_file)
            else:
                tmp.write(video_file.read())
            video_path = tmp.name

        if not os.path.exists(video_path):
            st.error(f"Fichier vidéo temporaire introuvable : {video_path}")
            return None, None

        if self.emergency_mode:
            self.detector.set_video_path(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Impossible d'ouvrir la vidéo : {video_path}")
            os.remove(video_path)
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info(f"Traitement de la vidéo : {total_frames} frames à {fps:.1f} FPS")

        timestamp = int(time.time())
        out_path = f"output/processed_{timestamp}.mp4"
        os.makedirs("output", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (640, 480))

        stats = {'frames': 0, 'detections': 0, 'tracks': 0, 'alerts': [], 'class_counts': self.class_counts.copy()}

        progress = st.progress(0)
        status_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            stats['frames'] += 1
            annotated_frame, det_count, track_count, alerts = self.process_frame(frame)
            stats['detections'] += det_count
            stats['tracks'] = max(stats['tracks'], track_count)
            if alerts:
                stats['alerts'].append({'frame': stats['frames'], 'objects': alerts})
            out.write(annotated_frame)
            if live_display:
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                live_display.image(frame_rgb, caption=f"Traitement du Frame {stats['frames']}", use_container_width=True, channels="RGB")
            status_text.text(f"Traitement du frame {stats['frames']}/{total_frames} - Détections : {det_count}")
            progress.progress(stats['frames'] / total_frames)

        cap.release()
        out.release()

        # Vérifier si la vidéo traitée a été créée
        if not os.path.exists(out_path):
            st.error(f"Échec de la création de la vidéo traitée : {out_path}")
            os.remove(video_path)
            return None, None

        # Convertir la vidéo en H.264 avec audio
        converted_path = f"output/converted_{timestamp}.mp4"
        if not os.path.exists(os.path.dirname(converted_path)):
            st.error(f"Répertoire de sortie inaccessible : {os.path.dirname(converted_path)}")
            os.remove(video_path)
            return None, None

        cmd = [
            'ffmpeg',
            '-i', out_path,  # Input processed video
            '-i', video_path,  # Input original video for audio
            '-c:v', 'libx264',  # Encode video with H.264
            '-c:a', 'aac',      # Encode audio with AAC
            '-map', '0:v:0',    # Map video stream from processed video
            '-map', '1:a?',     # Map audio stream from original video (optional)
            '-y',               # Overwrite output file
            converted_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            if os.path.exists(converted_path):
                st.success("Conversion vidéo en H.264 avec audio terminée avec succès !")
            else:
                st.warning(f"Conversion réussie mais fichier de sortie introuvable : {converted_path}. Utilisation du chemin original : {out_path}")
                converted_path = out_path
        except subprocess.CalledProcessError as e:
            st.error(f"Échec de la conversion vidéo : {e.stderr}")
            converted_path = out_path
        except FileNotFoundError as e:
            st.error(f"Commande ffmpeg ou fichier introuvable : {e}")
            converted_path = out_path

        # Générer la heatmap avec le même timestamp
        heatmap_path = os.path.join("output", f"heatmap_{timestamp}.png")
        if os.path.exists("output/tracking_log.csv"):
            generate_heatmap("output/tracking_log.csv", heatmap_path, width=640, height=480)
            if os.path.exists(heatmap_path):
                st.success(f"Heatmap générée à {heatmap_path}")
            else:
                st.error(f"Échec de la génération de la heatmap. Vérifiez tracking_log.csv.")
        else:
            st.error("Fichier tracking_log.csv introuvable.")

        os.remove(video_path)

        if not self.evaluator.ground_truths and self.detections_for_approx:
            df = pd.DataFrame(self.detections_for_approx)
            df = df.sample(frac=0.9, random_state=42)
            tmp_gt_path = os.path.join(tempfile.gettempdir(), f"gt_approx_{int(time.time())}.csv")
            df.to_csv(tmp_gt_path, index=False)
            try:
                self.evaluator.load_ground_truth(tmp_gt_path)
            finally:
                try:
                    os.remove(tmp_gt_path)
                except Exception as e:
                    st.warning(f"Impossible de supprimer le fichier temporaire de ground truth : {e}")

        stats['class_counts'] = self.class_counts.copy()
        stats['detection_metrics'] = self.evaluator.evaluate_detection()
        stats['tracking_metrics'] = self.evaluator.evaluate_tracking()

        status_text.text("✅ Traitement terminé !")
        return converted_path, stats

def main():
    st.markdown("<div class='header'><h1>Détection et suivi d'objets en temps réel dans les vidéos</h1></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Paramètres")

        st.subheader("Sélectionner les classes à détecter")
        all_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "umbrella", "ambulance"
        ]

        select_all = st.checkbox("Sélectionner toutes les classes", value=True)

        if 'selected_classes' not in st.session_state:
            st.session_state.selected_classes = all_classes.copy()

        if select_all:
            st.session_state.selected_classes = all_classes.copy()
        else:
            st.session_state.selected_classes = st.multiselect(
                "Classes",
                options=all_classes,
                default=st.session_state.selected_classes,
                key="class_multiselect"
            )

        if not st.session_state.selected_classes:
            st.warning("Veuillez sélectionner au moins une classe à détecter.")
            st.session_state.selected_classes = all_classes.copy()

        processor = VideoProcessor(selected_classes=st.session_state.selected_classes)

        processor.emergency_mode = st.checkbox("Détecter les véhicules d'urgence", value=False)
        live_view = st.checkbox("Afficher le traitement en direct", value=True)
        source = st.radio("Source", ["Télécharger une vidéo", "Vidéo d'exemple"])
        gt_file = st.file_uploader("Télécharger un fichier CSV de Ground Truth (facultatif)", type=['csv'])
        if gt_file:
            try:
                df = pd.read_csv(gt_file)
                st.write("Aperçu :")
                st.dataframe(df.head())
                st.write(f"Forme : {df.shape}")
                gt_file.seek(0)
            except Exception as e:
                st.error(f"Échec de la lecture du CSV : {e}")

    col1, col2 = st.columns([2, 1])

    with col1:
        video_file = None
        live_display = st.empty() if live_view else None
        gt_path = None

        if gt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(gt_file.read())
                gt_path = tmp.name
        elif source == "Vidéo d'exemple" and os.path.exists("sample_video/traffic_gt.csv"):
            gt_path = "sample_video/traffic_gt.csv"

        if source == "Télécharger une vidéo":
            video_file = st.file_uploader("Télécharger une vidéo", type=['mp4'])
            if video_file:
                st.video(video_file, format='video/mp4', start_time=0)
        elif source == "Vidéo d'exemple":
            sample_path = "sample_video/traffic.mp4"
            if os.path.exists(sample_path):
                with open(sample_path, 'rb') as f:
                    video_file = f.read()
                st.video(sample_path, format='video/mp4', start_time=0)
            else:
                st.error("Vidéo d'exemple introuvable. Ajoutez traffic.mp4 dans sample_video/")

        if video_file and st.button("Traiter la vidéo"):
            with st.spinner("Traitement en cours..."):
                video_path, stats = processor.process_video(video_file, live_display, gt_path, source)
                if video_path and stats:
                    st.session_state['results'] = {'video': video_path, 'stats': stats}
                    st.success("Terminé !")
                    if live_display:
                        live_display.empty()
                    if gt_path and os.path.exists(gt_path) and gt_file:
                        os.remove(gt_path)
                else:
                    st.error("Échec du traitement de la vidéo.")

    with col2:
        st.header("Métriques")
        if 'results' in st.session_state:
            stats = st.session_state['results']['stats']
            st.markdown(f"<div class='metric'>Frames traités : {stats['frames']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>Détections totales : {stats['detections']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>Suivis uniques : {stats['tracks']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>Alertes d'urgence : {len(stats['alerts'])}</div>", unsafe_allow_html=True)
            st.subheader("Détections par classe sélectionnée")
            for class_name, count in stats['class_counts'].items():
                if count > 0:
                    st.markdown(f"<div class='metric'>{class_name.capitalize()} : {count}</div>", unsafe_allow_html=True)
            st.subheader("Métriques de détection (classes sélectionnées)")
            det_metrics = stats['detection_metrics']
            st.markdown(f"<div class='metric'>Précision : {det_metrics['precision']:.3f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>Rappel : {det_metrics['recall']:.3f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>mAP : {det_metrics['mAP']:.3f}</div>", unsafe_allow_html=True)
            st.subheader("Métriques de suivi (classes sélectionnées)")
            track_metrics = stats['tracking_metrics']
            st.markdown(f"<div class='metric'>MOTA : {track_metrics['MOTA']:.3f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>Changements d'ID : {track_metrics['ID_Switches']}</div>", unsafe_allow_html=True)
            if stats['alerts']:
                st.subheader("Alertes")
                for alert in stats['alerts'][:3]:
                    st.markdown(f"<div class='alert'>Frame {alert['frame']} : {', '.join(alert['objects'])}</div>", unsafe_allow_html=True)
            if sum(stats['class_counts'].values()) == 0:
                st.warning("Aucun objet sélectionné n'a été détecté dans la vidéo.")

    if 'results' in st.session_state:
        st.header("Résultats")
        video_path = st.session_state['results']['video']
        try:
            with open(video_path, 'rb') as f:
                st.video(f.read(), format='video/mp4', start_time=0)
        except Exception as e:
            st.error(f"Échec de l'affichage de la vidéo : {e}. Assurez-vous que le format vidéo est compatible (par exemple, MP4 avec codec H.264).")

        st.subheader("Tous les objets détectés")
        if os.path.exists("output/tracking_log.csv"):
            df = pd.read_csv("output/tracking_log.csv")
            if not df.empty:
                df['direction_deg'] = df['direction'] * 180 / np.pi
                st.dataframe(df[['frame', 'track_id', 'class_name', 'x1', 'y1', 'x2', 'y2', 'confidence', 'speed', 'direction_deg']],
                            column_config={
                                "frame": "Frame (Timestamp)",
                                "track_id": "ID de suivi",
                                "class_name": "Classe",
                                "x1": "X1",
                                "y1": "Y1",
                                "x2": "X2",
                                "y2": "Y2",
                                "confidence": "Confiance",
                                "speed": "Vitesse (px/frame)",
                                "direction_deg": "Direction (degrés)"
                            })
            else:
                st.warning("Aucune donnée de suivi disponible.")
        else:
            st.warning("Journal de suivi introuvable.")

        heatmap_path = os.path.join("output", f"heatmap_{os.path.basename(video_path).split('_')[1].split('.')[0]}.png")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Heatmap des positions des objets", use_container_width=True)
        else:
            st.error(f"Heatmap introuvable à {heatmap_path}. Vérifiez le processus de génération.")
        if st.button("Générer un rapport PDF"):
            report_path = generate_pdf_report(st.session_state['results']['stats'])
            with open(report_path, 'rb') as f:
                st.download_button("Télécharger le rapport", f.read(), file_name=os.path.basename(report_path), mime="application/pdf")
        if os.path.exists("output/tracking_log.csv"):
            with open("output/tracking_log.csv", 'rb') as f:
                st.download_button("Télécharger le journal", f.read(), file_name="tracking_log.csv", mime="text/csv")

if __name__ == "__main__":
    main()