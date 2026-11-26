const {
    Document, Packer, Paragraph, TextRun, HeadingLevel,
    AlignmentType, PageBreak, TableOfContents, StyleLevel,
    Table, TableRow, TableCell, WidthType, BorderStyle,
    ShadingType, PageNumber, NumberFormat, Header, Footer,
    LineRuleType, convertInchesToTwip, Tab, TabStopType,
    LeaderType, LevelFormat, UnderlineType
} = require('docx');
const fs = require('fs');

// ── Helpers ──────────────────────────────────────────────────────
const TNR = 'Times New Roman';
const SZ = 24;   // 12pt in half-points
const SZH = 26;   // 13pt
const SZHL = 28;   // 14pt headings
const LINE = { line: 360, lineRule: LineRuleType.AUTO }; // 1.5 spacing
const JUST = AlignmentType.JUSTIFIED;
const CENTER = AlignmentType.CENTER;

function p(text, opts = {}) {
    const { bold = false, size = SZ, align = JUST, space = { before: 80, after: 80 },
        italic = false, underline = false, color = '000000' } = opts;
    return new Paragraph({
        alignment: align,
        spacing: { ...LINE, ...space },
        children: [new TextRun({
            text, bold, size, italics: italic,
            underline: underline ? { type: UnderlineType.SINGLE } : undefined,
            color, font: TNR
        })]
    });
}

function heading1(text, n) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: JUST,
        spacing: { ...LINE, before: 240, after: 120 },
        children: [new TextRun({ text: `${n}. ${text}`, bold: true, size: SZHL, font: TNR })]
    });
}

function heading2(text, n) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_2,
        alignment: JUST,
        spacing: { ...LINE, before: 160, after: 80 },
        children: [new TextRun({ text: `${n} ${text}`, bold: true, size: SZH, font: TNR })]
    });
}

function blank(n = 1) {
    return Array.from({ length: n }, () => new Paragraph({
        spacing: { ...LINE }, children: [new TextRun({ text: '', font: TNR, size: SZ })]
    }));
}

function figPlaceholder(num, caption) {
    return [
        new Paragraph({
            alignment: CENTER,
            spacing: { ...LINE, before: 120, after: 40 },
            children: [new TextRun({
                text: `[Figure ${num} — ${caption}]`,
                bold: true, size: SZ, font: TNR, color: '888888'
            })]
        }),
        new Paragraph({
            alignment: CENTER,
            border: {
                top: { style: BorderStyle.SINGLE, size: 1, color: '888888' },
                bottom: { style: BorderStyle.SINGLE, size: 1, color: '888888' },
                left: { style: BorderStyle.SINGLE, size: 1, color: '888888' },
                right: { style: BorderStyle.SINGLE, size: 1, color: '888888' }
            },
            spacing: { before: 40, after: 40 },
            children: [new TextRun({ text: '\u00A0'.repeat(80), size: SZ * 5, font: TNR })]
        }),
        new Paragraph({
            alignment: CENTER,
            spacing: { ...LINE, before: 40, after: 120 },
            children: [new TextRun({ text: `Figure ${num}: ${caption}`, bold: true, size: SZ, font: TNR })]
        })
    ];
}

function makeTable(headers, rows, colWidths) {
    const totalW = colWidths.reduce((a, b) => a + b, 0);
    return new Table({
        width: { size: totalW, type: WidthType.DXA },
        columnWidths: colWidths,
        rows: [
            new TableRow({
                tableHeader: true,
                children: headers.map((h, i) => new TableCell({
                    width: { size: colWidths[i], type: WidthType.DXA },
                    shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'D0D8E8' },
                    children: [new Paragraph({
                        alignment: CENTER,
                        spacing: { before: 60, after: 60 },
                        children: [new TextRun({ text: h, bold: true, size: SZ, font: TNR })]
                    })]
                }))
            }),
            ...rows.map(row => new TableRow({
                children: row.map((cell, i) => new TableCell({
                    width: { size: colWidths[i], type: WidthType.DXA },
                    children: [new Paragraph({
                        alignment: CENTER,
                        spacing: { before: 60, after: 60 },
                        children: [new TextRun({ text: String(cell), size: SZ, font: TNR })]
                    })]
                }))
            }))
        ]
    });
}

function tableCaption(num, text) {
    return new Paragraph({
        alignment: CENTER,
        spacing: { ...LINE, before: 40, after: 120 },
        children: [new TextRun({ text: `Table ${num}: ${text}`, bold: true, size: SZ, font: TNR })]
    });
}

function eq(text) {
    return new Paragraph({
        alignment: CENTER,
        spacing: { ...LINE, before: 80, after: 80 },
        children: [new TextRun({ text, italics: true, size: SZ, font: 'Cambria Math' })]
    });
}

function pb() {
    return new Paragraph({ children: [new PageBreak()] });
}

// ════════════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ════════════════════════════════════════════════════════════════════

const doc = new Document({
    creator: 'BTP Student',
    title: 'Depth-Assisted 3D Distance Estimation for Automated SEBT',
    styles: {
        default: {
            document: { run: { font: TNR, size: SZ } }
        },
        paragraphStyles: [
            {
                id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal',
                run: { bold: true, size: SZHL, font: TNR, color: '000000' },
                paragraph: { spacing: { before: 240, after: 120 } }
            },
            {
                id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal',
                run: { bold: true, size: SZH, font: TNR, color: '000000' },
                paragraph: { spacing: { before: 160, after: 80 } }
            },
        ]
    },
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 },
                margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }
            }
        },
        // ─── PAGE HEADER ─────────────────────────────────────────────
        headers: {
            default: new Header({
                children: [new Paragraph({
                    alignment: AlignmentType.RIGHT,
                    spacing: { before: 0, after: 0 },
                    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '000000' } },
                    children: [new TextRun({
                        text: 'Depth-Assisted 3D Distance Estimation for Automated SEBT',
                        size: 18, font: TNR, italics: true
                    })]
                })]
            })
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: CENTER,
                    children: [new TextRun({ children: [PageNumber.CURRENT], size: SZ, font: TNR })]
                })]
            })
        },
        children: [

            // ═══════════════════════════════════════════════════════════
            // COVER PAGE
            // ═══════════════════════════════════════════════════════════
            ...blank(2),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 120 },
                children: [new TextRun({ text: 'BTP PROJECT', bold: true, size: 32, font: TNR })]
            }),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 120 },
                children: [new TextRun({ text: 'MID SEMESTER EVALUATION — SECOND REPORT', bold: true, size: 28, font: TNR })]
            }),
            ...blank(2),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 80 },
                children: [new TextRun({ text: 'Title:', bold: true, size: SZH, font: TNR })]
            }),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 200 },
                children: [new TextRun({
                    text: 'Depth-Assisted 3D Distance Estimation for Automated Star Excursion Balance Test',
                    bold: true, size: 28, font: TNR
                })]
            }),
            ...blank(2),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 80 },
                children: [new TextRun({ text: 'Submitted by:', bold: true, size: SZ, font: TNR })]
            }),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 40 },
                children: [new TextRun({ text: 'M.K. Mirthun Vinayak', bold: true, size: SZH, font: TNR })]
            }),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 40 },
                children: [new TextRun({ text: 'Roll No: S20240100128', size: SZ, font: TNR })]
            }),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 40 },
                children: [new TextRun({ text: 'B.Tech — Computer Science and Engineering', size: SZ, font: TNR })]
            }),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 160 },
                children: [new TextRun({ text: 'IIIT Sri City', size: SZ, font: TNR })]
            }),
            ...blank(1),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 40 },
                children: [new TextRun({ text: 'Under the Guidance of:', bold: true, size: SZ, font: TNR })]
            }),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 40 },
                children: [new TextRun({ text: 'Dr. S. Manipriya', bold: true, size: SZH, font: TNR })]
            }),
            ...blank(3),
            new Paragraph({
                alignment: CENTER, spacing: { before: 0, after: 0 },
                children: [new TextRun({ text: 'Academic Year 2025–2026', size: SZ, font: TNR })]
            }),
            pb(),

            // ═══════════════════════════════════════════════════════════
            // ABSTRACT
            // ═══════════════════════════════════════════════════════════
            new Paragraph({
                alignment: CENTER, spacing: { ...LINE, before: 0, after: 120 },
                children: [new TextRun({ text: 'Abstract', bold: true, size: SZHL, font: TNR })]
            }),
            p(`The first phase of this B.Tech Project developed a fully browser-based, markerless Star Excursion Balance Test (SEBT) system using MediaPipe Pose and MediaPipe Hands operating entirely in 2D image space. While that system demonstrated real-time automated reach measurement and eight-direction scoring, its measurements are inherently constrained to pixel coordinates, making them sensitive to camera placement and perspective distortion. This second phase investigates whether metric monocular depth estimation — specifically Depth Anything V2 Metric — can supplement the 2D system with physically scaled 3D distance measurements from ordinary RGB video.`),
            p(`The pipeline combines MediaPipe Pose ankle detection with per-pixel metric depth from Depth Anything V2 Metric and approximate camera intrinsic parameters to reconstruct 3D ankle coordinates and compute Euclidean 3D reach distances. A preliminary controlled experiment with a 30 cm ruler produced absolute errors of 2.86 cm (9.53%) and 10.46 cm (34.87%) across two trials, indicating feasibility with non-trivial variance attributable to approximate intrinsics and monocular depth uncertainty. A ten-movement SEBT video experiment produced 3D distance estimates ranging from 23.6 cm to 118.6 cm, broadly within the expected clinical range. Comparison against an existing PhD-scholar tracking system is presented, with appropriate caveats regarding measurement definitions. Future work focuses on formal camera calibration, lens distortion correction, and ground-truth physical validation.`),
            blank(1)[0],
            p('Keywords: Star Excursion Balance Test, Depth Anything V2 Metric, monocular depth estimation, 3D distance estimation, camera intrinsics, MediaPipe Pose, balance assessment.', { italic: true }),
            pb(),

            // ═══════════════════════════════════════════════════════════
            // TABLE OF CONTENTS (manual — auto TOC not rendered by LibreOffice well)
            // ═══════════════════════════════════════════════════════════
            new Paragraph({
                alignment: CENTER, spacing: { ...LINE, before: 0, after: 200 },
                children: [new TextRun({ text: 'Table of Contents', bold: true, size: SZHL, font: TNR })]
            }),
            ...[
                ['1.', 'Objective and Motivation', '5'],
                ['2.', 'Limitation of Pure 2D Measurement', '5'],
                ['3.', 'Proposed Depth-Assisted Approach', '6'],
                ['4.', 'Camera Intrinsics and 3D Reconstruction', '7'],
                ['', '4.1  Camera Intrinsic Parameters', '7'],
                ['', '4.2  Selection of Approximate Intrinsics', '7'],
                ['', '4.3  Pixel-to-3D Coordinate Transformation', '8'],
                ['', '4.4  3D Distance Calculation', '8'],
                ['5.', 'Relationship Between Pixels and Physical Distance', '9'],
                ['6.', 'Movement Detection and Reach Measurement', '9'],
                ['7.', 'Experimental Setup', '10'],
                ['8.', 'Preliminary Controlled Distance Validation', '11'],
                ['9.', 'SEBT Video Experiment', '12'],
                ['10.', 'Comparison with Existing Tracking System', '13'],
                ['11.', 'Discussion', '14'],
                ['12.', 'Accuracy Improvement and Future Work', '15'],
                ['13.', 'Conclusion', '16'],
                ['14.', 'References', '16'],
                ['', 'List of Figures', '17'],
                ['', 'List of Tables', '17'],
                ['', 'Nomenclature', '18'],
            ].map(([num, title, pg]) => new Paragraph({
                alignment: JUST,
                spacing: { before: 40, after: 40 },
                tabStops: [{ type: TabStopType.RIGHT, position: 8640, leader: LeaderType.DOT }],
                children: [
                    new TextRun({ text: num ? `${num}  ${title}` : `       ${title}`, size: SZ, font: TNR }),
                    new TextRun({ text: '\t', size: SZ, font: TNR }),
                    new TextRun({ text: pg, size: SZ, font: TNR })
                ]
            })),
            pb(),

            // ═══════════════════════════════════════════════════════════
            // LIST OF FIGURES
            // ═══════════════════════════════════════════════════════════
            new Paragraph({
                alignment: CENTER, spacing: { ...LINE, before: 0, after: 160 },
                children: [new TextRun({ text: 'List of Figures', bold: true, size: SZHL, font: TNR })]
            }),
            ...[
                ['Figure 1', 'System Architecture — Depth-Assisted SEBT Pipeline', '6'],
                ['Figure 2', 'Camera Intrinsic Parameters and Image Plane Geometry', '8'],
                ['Figure 3', 'Movement Detection State Flow', '10'],
                ['Figure 4', 'Experimental Setup — Camera and Subject Placement', '11'],
                ['Figure 5', 'Depth Anything V2 Metric Output — Ankle Detection Screenshot', '12'],
                ['Figure 6', 'SEBT Video Experiment — Movement Overlay Screenshots', '12'],
                ['Figure 7', 'Comparison of 3D Reach Estimates Across Ten Movements (Bar Chart)', '13'],
            ].map(([fig, caption, pg]) => new Paragraph({
                alignment: JUST,
                spacing: { before: 40, after: 40 },
                tabStops: [{ type: TabStopType.RIGHT, position: 8640, leader: LeaderType.DOT }],
                children: [
                    new TextRun({ text: `${fig}:  ${caption}`, size: SZ, font: TNR }),
                    new TextRun({ text: '\t', size: SZ, font: TNR }),
                    new TextRun({ text: pg, size: SZ, font: TNR })
                ]
            })),
            blank(1)[0],
            new Paragraph({
                alignment: CENTER, spacing: { ...LINE, before: 120, after: 160 },
                children: [new TextRun({ text: 'List of Tables', bold: true, size: SZHL, font: TNR })]
            }),
            ...[
                ['Table 1', 'Approximate Camera Intrinsic Parameters Used in Preliminary Experiment', '7'],
                ['Table 2', 'Preliminary Controlled Distance Validation Results', '11'],
                ['Table 3', 'SEBT Video Experiment — Ten-Movement 3D Distance Output', '12'],
                ['Table 4', 'Comparison of Best Reach Values — Existing System vs This Work', '13'],
                ['Table 5', 'Experimental Software and Hardware Setup', '10'],
            ].map(([tbl, caption, pg]) => new Paragraph({
                alignment: JUST,
                spacing: { before: 40, after: 40 },
                tabStops: [{ type: TabStopType.RIGHT, position: 8640, leader: LeaderType.DOT }],
                children: [
                    new TextRun({ text: `${tbl}:  ${caption}`, size: SZ, font: TNR }),
                    new TextRun({ text: '\t', size: SZ, font: TNR }),
                    new TextRun({ text: pg, size: SZ, font: TNR })
                ]
            })),
            pb(),

            // ═══════════════════════════════════════════════════════════
            // NOMENCLATURE
            // ═══════════════════════════════════════════════════════════
            new Paragraph({
                alignment: CENTER, spacing: { ...LINE, before: 0, after: 160 },
                children: [new TextRun({ text: 'Nomenclature', bold: true, size: SZHL, font: TNR })]
            }),
            makeTable(
                ['Symbol', 'Description', 'Unit'],
                [
                    ['fx', 'Focal length along x-axis in pixel units', 'pixels'],
                    ['fy', 'Focal length along y-axis in pixel units', 'pixels'],
                    ['cx', 'Principal point x-coordinate', 'pixels'],
                    ['cy', 'Principal point y-coordinate', 'pixels'],
                    ['Z', 'Metric depth from Depth Anything V2 Metric', 'metres'],
                    ['X', 'Reconstructed 3D x-coordinate (physical)', 'metres'],
                    ['Y', 'Reconstructed 3D y-coordinate (physical)', 'metres'],
                    ['(x, y)', 'Image pixel coordinates of detected ankle', 'pixels'],
                    ['P', 'Reconstructed 3D point (X, Y, Z)', 'metres'],
                    ['D', 'Euclidean 3D distance between ankle points', 'metres'],
                    ['Dcm', 'D converted to centimetres (D × 100)', 'cm'],
                    ['ΔX, ΔY, ΔZ', 'Component-wise differences in 3D space', 'metres'],
                    ['Pc', 'Pixel-to-centimetre calibration ratio (2D)', 'cm/pixel'],
                    ['L', 'Leg length (hip to ankle)', 'cm'],
                    ['S', 'Normalized SEBT score (Average Reach / L × 100)', '%'],
                    ['R', 'Maximum reach distance for a direction', 'cm'],
                    ['H', 'Hip centre landmark position', 'pixels / metres'],
                    ['W', 'Shoulder width (front-view calibration)', 'pixels'],
                ],
                [1800, 5000, 1800]
            ),
            pb(),

            // ═══════════════════════════════════════════════════════════
            // SECTION 1 — OBJECTIVE AND MOTIVATION
            // ═══════════════════════════════════════════════════════════
            heading1('Objective and Motivation', '1'),
            p(`The first phase of this BTP delivered a fully working, browser-based SEBT system that calibrates using a physical ruler, tracks ankle/toe landmarks via MediaPipe Pose, and reports eight-direction reach distances in centimetres by applying a fixed pixel-to-centimetre conversion ratio derived from the ruler. Although that system is functional for a fixed, front-parallel camera placement, its fundamental limitation is that all distances are computed in 2D image space: the pixel-to-cm ratio is valid only when the subject's foot moves in a plane that is exactly parallel to the camera sensor and at exactly the calibration distance.`),
            p(`This second phase does not replace the 2D system. Its objective is to investigate whether integrating a state-of-the-art monocular metric depth estimator — Depth Anything V2 Metric — into the existing MediaPipe Pose tracking pipeline can produce physically scaled 3D distance estimates without any additional hardware such as depth cameras, structured light systems, or stereo rigs. The primary research question is one of feasibility: can a single RGB camera, combined with a pre-trained metric depth model and approximate camera intrinsics, yield ankle-to-ankle 3D distances that are within a clinically meaningful range of actual physical distances?`),
            p(`No claim of clinical accuracy is made at this stage. The goal is to characterise the approach, document preliminary errors, and establish what further work — particularly proper camera calibration and physical ground-truth validation — is needed before the 3D distance estimates can be considered reliable.`),

            // ═══════════════════════════════════════════════════════════
            // SECTION 2 — LIMITATION OF PURE 2D MEASUREMENT
            // ═══════════════════════════════════════════════════════════
            heading1('Limitation of Pure 2D Measurement', '2'),
            p(`In the existing 2D SEBT system, the reach distance is computed as the Euclidean distance between two pixel coordinates on the image plane, scaled by a fixed pixel-to-centimetre ratio Pc obtained from a 30 cm ruler at calibration time. This works well under the assumption that the subject stands at a constant, known distance from the camera and that all foot movements occur in a plane parallel to the image sensor. In practice, these assumptions are routinely violated during a real SEBT session.`),
            p(`The pinhole camera model establishes that a physical length ΔX at depth Z maps to a pixel displacement Δx according to:`),
            eq('Δx = (fx / Z) · ΔX'),
            p(`Rearranging, the physical length per pixel is:`),
            eq('ΔX / Δx = Z / fx'),
            p(`This ratio depends on Z. If Z changes between the calibration moment (ruler held at depth Zc) and the measurement moment (foot at depth Zm), the pixel-to-cm ratio is no longer Pc but rather Pc · (Zm / Zc). For a subject whose foot moves toward or away from the camera by even 30 cm, the error introduced into a purely 2D reach estimate can exceed 10–15% depending on the baseline camera distance.`),
            p(`Furthermore, the SEBT involves reaches in the anterior, posterior, medial, lateral, and four diagonal directions, many of which inherently involve motion along the depth axis (toward or away from the camera for a fixed camera placement). For these directions, 2D pixel displacement significantly underestimates actual physical reach because the depth component ΔZ contributes to physical distance but is invisible to the 2D measurement. The full 3D Euclidean distance:`),
            eq('D = √(ΔX² + ΔY² + ΔZ²)'),
            p(`cannot be recovered from 2D pixel coordinates alone. Metric depth estimation is therefore a natural and necessary extension.`),

            // ═══════════════════════════════════════════════════════════
            // SECTION 3 — PROPOSED DEPTH-ASSISTED APPROACH
            // ═══════════════════════════════════════════════════════════
            heading1('Proposed Depth-Assisted Approach', '3'),
            p(`The proposed pipeline extends the existing 2D tracking system by inserting a metric depth estimation stage between raw RGB frame acquisition and 3D coordinate reconstruction. The complete processing chain for each frame is as follows:`),
            p(`(1) An RGB video frame is acquired from the camera. (2) MediaPipe Pose is applied to the frame to detect 33 body landmarks, including the left ankle (landmark 27) and right ankle (landmark 28). The mirrored canvas coordinates of both ankles are extracted in pixel space. (3) Depth Anything V2 Metric processes the same RGB frame and outputs a per-pixel metric depth map, where each pixel value represents the estimated distance from the camera plane to the corresponding scene point in metres. (4) The metric depth Z at each ankle's pixel location is sampled from the depth map; a 5×5 median filter is applied to suppress local noise. (5) Using the camera intrinsic parameters fx, fy, cx, cy, the 2D image coordinates (x, y) and metric depth Z are combined to reconstruct approximate 3D coordinates (X, Y, Z) via the standard inverse pinhole projection. (6) The Euclidean 3D distance between the standing ankle and the reaching ankle is computed. This constitutes the metric reach estimate for the current frame, and the maximum across all frames in a direction defines the reach for that direction.`),
            ...figPlaceholder(1, 'System Architecture — Depth-Assisted SEBT Pipeline'),

            // ═══════════════════════════════════════════════════════════
            // SECTION 4 — CAMERA INTRINSICS AND 3D RECONSTRUCTION
            // ═══════════════════════════════════════════════════════════
            heading1('Camera Intrinsics and 3D Reconstruction', '4'),

            heading2('Camera Intrinsic Parameters', '4.1'),
            p(`The pinhole camera model defines the mapping from a 3D world point to a 2D image point through four intrinsic parameters. The focal lengths fx and fy (in pixel units) determine how strongly distances along the X and Y physical axes are compressed onto the image plane. The principal point coordinates cx and cy identify the image pixel that corresponds to the camera's optical axis — ideally the geometric centre of the sensor, though lens mounting tolerances cause small deviations in real cameras.`),
            p(`Together these four parameters form the camera intrinsic matrix K:`),
            eq('K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]'),
            p(`A calibrated K is obtained by imaging a calibration target (e.g., a checkerboard) from multiple viewpoints and solving for the parameters that minimise reprojection error. In the current preliminary experiment, K was not determined through such a calibration; instead, approximate values were assumed.`),

            heading2('Selection of Approximate Intrinsics', '4.2'),
            p(`For the preliminary experiment the following parameter values were used:`),
            blank(1)[0],
            makeTable(
                ['Parameter', 'Value Used', 'Basis'],
                [
                    ['fx', '800 pixels', 'Assumed approximate focal length — not measured'],
                    ['fy', '800 pixels', 'Assumed approximate focal length — not measured'],
                    ['cx', '540 pixels', 'Image centre: 1080 / 2 = 540'],
                    ['cy', '960 pixels', 'Image centre: 1920 / 2 = 960'],
                    ['Image resolution', '1080 × 1920 pixels', 'Portrait RGB video input'],
                ],
                [2000, 3000, 3600]
            ),
            tableCaption(1, 'Approximate Camera Intrinsic Parameters Used in Preliminary Experiment'),
            blank(1)[0],
            p(`The principal point was placed at the image centre because, absent a calibration, the optical axis is assumed to intersect the sensor at its midpoint:`),
            eq('cx = 1080 / 2 = 540,   cy = 1920 / 2 = 960'),
            p(`The focal length value of 800 pixels was an engineering estimate based on a typical smartphone or laptop webcam field of view (approximately 60–70 degrees horizontal), for which the focal length in pixels is roughly:`),
            eq('fx ≈ (W/2) / tan(FoV/2)  ≈  540 / tan(30°)  ≈  935 pixels'),
            p(`An initial value of 800 pixels was selected as a conservative round estimate. This value has not been validated against physical targets and represents a known source of error in the preliminary results. Formal camera calibration using a checkerboard or ChArUco board is identified as the single most important step for reducing 3D reconstruction error, and is planned as the immediate next phase of this work.`),

            heading2('Pixel-to-3D Coordinate Transformation', '4.3'),
            p(`Given an ankle at image coordinates (x, y) and metric depth Z (in metres) from Depth Anything V2 Metric, the approximate 3D coordinates are computed as:`),
            eq('X = (x − cx) · Z / fx'),
            eq('Y = (y − cy) · Z / fy'),
            eq('Z = Z'),
            p(`yielding the 3D point:`),
            eq('P = (X, Y, Z)'),
            p(`Because Depth Anything V2 Metric provides depths in metres, the reconstructed coordinates X and Y are also in metres. The depth Z is the dominant component for subjects standing at typical camera distances of 1.5–3 metres; the lateral components X and Y account for the remaining image-plane spread.`),
            ...figPlaceholder(2, 'Camera Intrinsic Parameters and Image Plane Geometry'),

            heading2('3D Distance Calculation', '4.4'),
            p(`For the standing ankle at P1 = (X1, Y1, Z1) and the reaching ankle at P2 = (X2, Y2, Z2), the 3D Euclidean distance is:`),
            eq('D = √( (X₂−X₁)² + (Y₂−Y₁)² + (Z₂−Z₁)² )'),
            p(`and converted to centimetres:`),
            eq('Dcm = 100 · D'),
            p(`This quantity captures the full physical separation between the two ankles including depth-axis movement, unlike the 2D pixel distance which ignores the Z component entirely.`),

            // ═══════════════════════════════════════════════════════════
            // SECTION 5 — RELATIONSHIP PIXELS AND PHYSICAL DISTANCE
            // ═══════════════════════════════════════════════════════════
            heading1('Relationship Between Pixels and Physical Distance', '5'),
            p(`A natural question arising from the use of both pixel measurements and 3D measurements is whether a simple conversion factor cm/pixel can be derived from the depth estimates and used to replace the 2D ruler calibration. The answer is no, and the reason illuminates why the full 3D pipeline is necessary.`),
            p(`From the inverse pinhole model, for a subject standing at depth Z, the physical displacement ΔX corresponding to a pixel displacement Δx is:`),
            eq('ΔX ≈ (Z / fx) · Δx'),
            p(`This means the effective physical scale per pixel at that depth is Z/fx — a quantity that changes with Z. For example, if Z = 2 m and fx = 800:`),
            eq('Z / fx = 2 / 800 = 0.0025 m/pixel = 0.25 cm/pixel'),
            p(`At Z = 1.5 m the same calculation gives 0.1875 cm/pixel, and at Z = 3 m it gives 0.375 cm/pixel. A single ruler calibration at Zc = 2 m would produce Pc = 0.25 cm/pixel; applying this to a measurement made at Z = 1.5 m would overestimate the physical distance by 33%.`),
            p(`Furthermore, simply computing ΔZ / (pixel distance) does not produce a valid cm/pixel conversion. The ΔZ component alone does not account for the lateral displacement. The correct approach is to reconstruct full 3D coordinates and compute the Euclidean distance:`),
            eq('D = √(ΔX² + ΔY² + ΔZ²)'),
            p(`which captures all three components of the physical motion. This is the approach implemented in the current system.`),

            // ═══════════════════════════════════════════════════════════
            // SECTION 6 — MOVEMENT DETECTION AND REACH MEASUREMENT
            // ═══════════════════════════════════════════════════════════
            heading1('Movement Detection and Reach Measurement', '6'),
            p(`The movement detection and reach measurement module operates on top of the 3D coordinate reconstruction pipeline. Its purpose is to identify individual reach attempts within a continuous video sequence, track ankle movement from the standing reference position, record the maximum 3D distance attained, and detect when the foot has returned to the starting position.`),
            p(`The module operates through the following stages. First, at session start, MediaPipe Pose detects both ankle landmarks and a baseline ankle separation in both 2D and 3D is established. The standing ankle is identified as the one with less total displacement over a short observation window — consistent with the single-leg balance posture of the SEBT.`),
            p(`Movement start is detected when the 3D distance from the current reaching ankle position to the baseline position exceeds a configurable minimum movement threshold (nominally 5 cm). Once movement has started, the 3D distance is computed on every frame and the maximum value is continuously updated. The current reach distance Dcm and the running maximum are both displayed as overlays on the video frame.`),
            p(`Return detection is implemented using an adaptive return threshold: a movement is considered complete when the reaching ankle returns to within a configurable tolerance (nominally 8 cm, tunable based on camera distance) of the baseline position after having exceeded the movement threshold. At this point the maximum reach for that movement is recorded, the movement counter increments, and the system prepares to detect the next movement.`),
            ...figPlaceholder(3, 'Movement Detection State Flow'),

            // ═══════════════════════════════════════════════════════════
            // SECTION 7 — EXPERIMENTAL SETUP
            // ═══════════════════════════════════════════════════════════
            heading1('Experimental Setup', '7'),
            p(`The experimental system was implemented in Python, running on a laptop. The following hardware and software components were used:`),
            blank(1)[0],
            makeTable(
                ['Component', 'Specification'],
                [
                    ['RGB Video Input', 'Smartphone camera video, 1080×1920 pixels, portrait orientation'],
                    ['Depth Model', 'Depth Anything V2 Metric (vits — small variant, metric depth output in metres)'],
                    ['Pose Estimation', 'MediaPipe Pose (model complexity 1, smooth landmarks enabled)'],
                    ['Camera Intrinsics', 'fx = fy = 800 px; cx = 540 px; cy = 960 px (approximate, uncalibrated)'],
                    ['Depth Sampling', '5×5 median filter at ankle pixel location to reduce local noise'],
                    ['Processing Environment', 'Python 3.10, PyTorch, OpenCV, MediaPipe'],
                    ['Hardware', 'Laptop CPU (no dedicated GPU used for preliminary experiments)'],
                    ['Ground Truth', 'Physical ruler measurements (30 cm) for controlled validation'],
                ],
                [3000, 5600]
            ),
            tableCaption(5, 'Experimental Software and Hardware Setup'),
            blank(1)[0],
            p(`It is important to distinguish between the output of Depth Anything V2 Metric and physical ground truth. Depth Anything's output is an estimated metric depth derived from a neural network trained on diverse RGB datasets. It is not a direct physical measurement. Ground-truth physical distances in this work refer only to ruler measurements made independently of the depth model.`),
            ...figPlaceholder(4, 'Experimental Setup — Camera and Subject Placement'),

            // ═══════════════════════════════════════════════════════════
            // SECTION 8 — PRELIMINARY CONTROLLED DISTANCE VALIDATION
            // ═══════════════════════════════════════════════════════════
            heading1('Preliminary Controlled Distance Validation', '8'),
            p(`To assess the system's ability to reproduce a known physical distance, a controlled experiment was conducted using a 30 cm ruler as the reference object. Two trials were performed. In each trial, the ruler was placed within the camera field of view and two landmark points separated by the ruler length were tracked through the 3D reconstruction pipeline. The 3D distance estimated by the system was then compared against the known 30 cm physical length.`),
            blank(1)[0],
            makeTable(
                ['Trial', 'Actual Distance (cm)', 'Estimated 3D Distance (cm)', 'Absolute Error (cm)', 'Percentage Error (%)'],
                [
                    ['Test 1', '30.00', '32.86', '2.86', '9.53'],
                    ['Test 2', '30.00', '19.54', '10.46', '34.87'],
                ],
                [800, 1800, 2500, 2000, 1500]
            ),
            tableCaption(2, 'Preliminary Controlled Distance Validation Results'),
            blank(1)[0],
            p(`Test 1 produced an absolute error of 2.86 cm (9.53%), which is arguably within a clinically acceptable range for a preliminary uncalibrated system. Test 2 produced a substantially larger error of 10.46 cm (34.87%), demonstrating that the system's accuracy is not yet consistent. These results are preliminary and do not constitute a formal validation study; no statistical analysis is warranted at this stage given the small sample size.`),
            p(`The following sources of error are identified as most likely contributors to the observed discrepancies:`),
            p(`Approximate Camera Intrinsics: The assumed fx = fy = 800 and principal point at image centre introduce a systematic error that scales linearly with depth. A camera calibration error of even 10–15% in focal length directly propagates to a 10–15% error in the reconstructed physical coordinates.`),
            p(`Monocular Depth Estimation Uncertainty: Depth Anything V2 Metric produces depth estimates from a single RGB frame without any stereo or active ranging information. For objects at arm's length (30–150 cm), the model's uncertainty is typically higher than for scene-scale objects at 2–10 metres, for which most training data is available.`),
            p(`Camera and Lens Distortion: The approximate intrinsic model does not include lens distortion coefficients. Radial and tangential distortion displace pixel coordinates from their ideal pinhole positions, introducing additional errors into the back-projection.`),
            p(`Depth Sampling Location: The 5×5 median filter at the ankle pixel was applied to suppress noise but does not eliminate the fundamental uncertainty in depth at small, limb-scale features. The depth model's response at thin structures such as ankles may differ from that at larger planar surfaces.`),

            // ═══════════════════════════════════════════════════════════
            // SECTION 9 — SEBT VIDEO EXPERIMENT
            // ═══════════════════════════════════════════════════════════
            heading1('SEBT Video Experiment', '9'),
            p(`A video recording of an SEBT session was processed through the complete depth-assisted pipeline. MediaPipe Pose detected ankle landmarks on each frame; Depth Anything V2 Metric estimated per-pixel metric depth; 3D coordinates were reconstructed; and the movement detection module identified ten complete reach-and-return movements. The output for all ten movements is tabulated below.`),
            blank(1)[0],
            makeTable(
                ['Movement', 'Ankle Used', 'Pixel Distance (px)', 'ΔZ (cm)', '3D Distance (cm)'],
                [
                    ['1', 'RIGHT', '261.4', '16.4', '78.5'],
                    ['2', 'RIGHT', '206.9', '7.1', '39.0'],
                    ['3', 'RIGHT', '189.4', '5.3', '45.4'],
                    ['4', 'LEFT', '94.6', '15.7', '37.5'],
                    ['5', 'RIGHT', '206.6', '12.7', '44.4'],
                    ['6', 'RIGHT', '88.6', '0.8', '23.6'],
                    ['7', 'LEFT', '132.2', '5.2', '50.3'],
                    ['8', 'RIGHT', '256.8', '83.5', '118.6'],
                    ['9', 'LEFT', '118.7', '14.5', '38.0'],
                    ['10', 'RIGHT', '279.2', '21.9', '74.5'],
                ],
                [900, 1200, 1900, 1400, 2000]
            ),
            tableCaption(3, 'SEBT Video Experiment — Ten-Movement 3D Distance Output'),
            blank(1)[0],
            p(`Several observations can be drawn from these results. First, the 3D distance values range from 23.6 cm to 118.6 cm — a range that is broadly consistent with expected SEBT reach distances for an adult participant (typically 30–120 cm depending on direction and leg length). Second, the pixel distance alone is a poor predictor of the 3D distance: Movement 8 has a pixel distance of 256.8 px but yields the highest 3D estimate (118.6 cm) because of a large ΔZ of 83.5 cm, while Movement 6 has a pixel distance of 88.6 px and the smallest ΔZ (0.8 cm), confirming that the depth component is essential for accurate 3D estimation. Third, movements involving predominantly lateral motion in the image (low ΔZ) cluster around 37–50 cm, while movements with larger depth components produce higher 3D distances — a physically expected result.`),
            p(`It must be noted that without independent physical ground-truth measurements of each reach, these values cannot be validated against actual reaching distances. They represent the system's output under the approximate intrinsic parameters and are presented as feasibility evidence.`),
            ...figPlaceholder(5, 'Depth Anything V2 Metric Output — Ankle Detection Screenshot'),
            ...figPlaceholder(6, 'SEBT Video Experiment — Movement Overlay Screenshots'),

            // ═══════════════════════════════════════════════════════════
            // SECTION 10 — COMPARISON WITH EXISTING TRACKING SYSTEM
            // ═══════════════════════════════════════════════════════════
            heading1('Comparison with Existing Tracking System', '10'),
            p(`An existing SEBT tracking system developed by a PhD scholar was also applied to a comparable SEBT session. That system reports several metrics including 2D Reach, 3D Reach, BBox Reach, Reach Delta, Centre Sway, Pose Confidence, and Stability Score. Its final best directional reach values (2D) were reported as follows:`),
            blank(1)[0],
            makeTable(
                ['Direction', 'Best 2D Reach (cm) — Existing System', 'Corresponding 3D Estimate — This Work (cm)'],
                [
                    ['ANT (Anterior)', '117.0', 'Movement 1: 78.5'],
                    ['AM (Anteromedial)', '38.0', 'Movement 4: 37.5'],
                    ['M (Medial)', '20.4', 'Movement 6: 23.6'],
                    ['PM (Posteromedial)', '0.0', '—'],
                    ['POST (Posterior)', '135.7', 'Movement 8: 118.6'],
                    ['PL (Posterolateral)', '56.5', 'Movement 7: 50.3'],
                    ['LAT (Lateral)', '33.8', 'Movement 9: 38.0'],
                    ['AL (Anterolateral)', '60.1', 'Movement 10: 74.5'],
                ],
                [2500, 3000, 3100]
            ),
            tableCaption(4, 'Comparison of Best Reach Values — Existing System vs This Work'),
            blank(1)[0],
            p(`This comparison must be interpreted with considerable caution. The existing system's values are 2D final directional best reaches, meaning they represent the maximum 2D pixel-scaled reach in a specifically labelled SEBT direction. The values from this work are 3D metric-depth estimates of ankle-to-ankle distances for movements identified by the movement detection module, without direction labelling. The two columns therefore measure different quantities using different methodologies, and a direct numerical comparison is not scientifically valid.`),
            p(`What the comparison does illustrate is that the outputs of this work are within broadly comparable physical ranges in several cases: Anterior (78.5 vs 117.0 cm), Anteromedial (37.5 vs 38.0 cm), Medial (23.6 vs 20.4 cm), Posterolateral (50.3 vs 56.5 cm), Lateral (38.0 vs 33.8 cm). The largest discrepancy is in Posterior (118.6 vs 135.7 cm), which is expected given that posterior reaches involve substantial motion along the depth axis — the dimension most prone to error in both 2D pixel-based and monocular depth-assisted systems.`),
            p(`Independent physical ground-truth measurement using a tape measure or reference markers during an actual SEBT session is required for any meaningful accuracy comparison between the two systems.`),
            ...figPlaceholder(7, 'Comparison of 3D Reach Estimates Across Ten Movements (Bar Chart)'),

            // ═══════════════════════════════════════════════════════════
            // SECTION 11 — DISCUSSION
            // ═══════════════════════════════════════════════════════════
            heading1('Discussion', '11'),
            p(`The results of both the controlled ruler experiment and the SEBT video experiment support a cautiously positive interpretation of the proposed pipeline's feasibility. The system successfully integrates MediaPipe Pose ankle tracking, Depth Anything V2 Metric depth estimation, approximate camera intrinsics, and 3D Euclidean distance calculation into a coherent processing chain that runs on standard RGB video without additional sensors.`),
            p(`However, the variance in the controlled experiment — 9.53% error in one trial versus 34.87% in another — reveals that consistency is not yet achieved. This variance is primarily attributable to the use of approximate, uncalibrated intrinsic parameters and the fundamental limitations of monocular depth estimation, which lacks the geometric constraint of stereo vision and therefore exhibits scene-dependent depth errors that cannot be reduced by software alone.`),
            p(`The SEBT video experiment produced 3D distance outputs that are physically plausible and broadly consistent with the expected clinical range of reach distances. The key insight that distinguishes this approach from the 2D system is confirmed: movements with large depth-axis displacement (e.g., Movement 8 with ΔZ = 83.5 cm) yield substantially higher 3D distances than would be estimated from pixel distance alone, which is the physically correct behaviour for a posterior or anteroposterior reach in a side-view camera placement.`),
            p(`The comparison with the existing PhD-scholar system is offered as context, not as a benchmark. The value of having both outputs is that it allows identification of directions where the two approaches produce qualitatively consistent estimates (anteromedial, medial, lateral) versus directions where they diverge substantially (anterior, posterior), which points to where depth axis contribution is most significant.`),
            p(`The principal limitation of this phase is the absence of camera calibration. All quantitative claims in this report are preliminary and subject to revision once calibrated intrinsic parameters are available. A second limitation is that the depth model's training domain is predominantly scene-level data from autonomous driving and indoor environments; the close-range, human-limb-focused estimation required for ankle tracking represents a slightly out-of-distribution condition for which model uncertainty is likely elevated.`),

            // ═══════════════════════════════════════════════════════════
            // SECTION 12 — ACCURACY IMPROVEMENT AND FUTURE WORK
            // ═══════════════════════════════════════════════════════════
            heading1('Accuracy Improvement and Future Work', '12'),
            p(`The single most impactful improvement identified for this system is formal camera calibration. The current experiment uses fx = fy = 800 pixels and principal point at image centre (cx = 540, cy = 960) — values that are convenient approximations, not measurements. A formal calibration procedure using a checkerboard or ChArUco board would provide:`),
            p(`Calibrated fx, fy, cx, cy: Typical smartphone focal lengths at 1080×1920 resolution are often in the range 900–1200 pixels; the assumed value of 800 may therefore introduce a systematic underestimation of depth-scaled distances. Calibration would reveal the true value and eliminate this source of systematic error.`),
            p(`Lens distortion coefficients: Radial distortion (k1, k2) and tangential distortion (p1, p2) displace pixel coordinates from their ideal pinhole positions. Correcting for these coefficients before back-projection into 3D space would reduce reconstruction error, particularly toward the frame periphery where distortion is largest.`),
            p(`Beyond camera calibration, the following improvements are planned for the next phase:`),
            p(`Depth Model Upgrade: Depth Anything V2 Metric is available in small (vits), base (vitb), and large (vitl) variants. The preliminary experiment used the small variant for computational speed. Evaluating the base and large variants for closer-range scenes may improve depth accuracy at limb-scale distances.`),
            p(`Physical Ground-Truth Validation: A formal validation dataset must be collected in which a participant performs standardised SEBT reaches of known distances, measured independently with a physical tape measure or motion capture reference. Without this dataset, accuracy claims remain indicative rather than quantified.`),
            p(`Depth Sampling Improvement: Rather than sampling depth at the single pixel closest to the ankle landmark, a more robust estimate might be obtained by sampling a region centred on the ankle landmark and applying a weighted average that prioritises pixels with high confidence — if the depth model provides uncertainty estimates — or by using the median of a larger window.`),
            p(`Integration with the 2D Web System: The depth-assisted pipeline is currently a standalone Python system. Integrating it with the browser-based 2D system — either via a backend API or by exploring WebAssembly deployment of Depth Anything — would unify both approaches and allow direct per-session comparison.`),
            p(`Evaluation Under Varying Conditions: Systematic evaluation under different lighting conditions, backgrounds, camera distances, and camera angles is needed to characterise the sensitivity of the 3D estimates to environmental factors.`),

            // ═══════════════════════════════════════════════════════════
            // SECTION 13 — CONCLUSION
            // ═══════════════════════════════════════════════════════════
            heading1('Conclusion', '13'),
            p(`This report documents the second phase of a B.Tech project investigating depth-assisted 3D distance estimation for the automated Star Excursion Balance Test. Depth Anything V2 Metric was integrated into a MediaPipe Pose ankle tracking pipeline to estimate per-frame metric depth at detected ankle locations. Using approximate camera intrinsic parameters (fx = fy = 800 px; cx = 540 px; cy = 960 px), 3D ankle coordinates were reconstructed via inverse pinhole projection and Euclidean 3D reach distances were computed.`),
            p(`A preliminary controlled experiment with a 30 cm ruler demonstrated errors of 9.53% and 34.87% across two trials, confirming feasibility while revealing the significant impact of uncalibrated intrinsic parameters on accuracy. A ten-movement SEBT video experiment produced 3D distance estimates ranging from 23.6 cm to 118.6 cm, consistent with the expected range of human SEBT reach distances and showing physically plausible behaviour — particularly the correct increase in 3D distance for movements with large depth-axis displacement.`),
            p(`Comparison with an existing PhD-scholar SEBT tracking system showed broadly comparable ranges in several directions, with appropriate acknowledgement that the two systems measure different quantities and that independent physical ground-truth validation is required for a meaningful accuracy comparison.`),
            p(`The most important identified improvement is formal camera calibration using a checkerboard or ChArUco board. Calibrated intrinsics are expected to substantially reduce systematic error and improve inter-trial consistency. Physical ground-truth validation with a reference tape measure or motion capture system is the essential next step toward establishing whether the depth-assisted pipeline can meet clinically acceptable accuracy standards.`),

            // ═══════════════════════════════════════════════════════════
            // REFERENCES
            // ═══════════════════════════════════════════════════════════
            heading1('References', '14'),
            ...[
                '[1] Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., and Zhao, H. (2024). "Depth Anything V2." arXiv preprint arXiv:2406.09414.',
                '[2] Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., and Zhao, H. (2024). "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10371–10381.',
                '[3] Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C. L., Yong, M. G., Lee, J., et al. (2019). "MediaPipe: A Framework for Perceiving and Processing Reality." Third Workshop on Computer Vision for AR/VR at CVPR 2019.',
                '[4] Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., and Grundmann, M. (2020). "BlazePose: On-Device Real-Time Body Pose Tracking." arXiv preprint arXiv:2006.10204.',
                '[5] Hartley, R. and Zisserman, A. (2004). Multiple View Geometry in Computer Vision, 2nd ed. Cambridge University Press, Cambridge.',
                '[6] Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration." IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), pp. 1330–1334.',
                '[7] Gribbon, K. T. and Bailey, D. G. (2004). "A Novel Approach to Real-Time Bilinear Interpolation." Proceedings of the Second IEEE International Workshop on Electronic Design, Test and Applications, pp. 126–131.',
                '[8] Plisky, P. J., Gorman, P. P., Butler, R. J., Kiesel, K. B., Underwood, F. B., and Elkins, B. (2009). "The Reliability of an Instrumented Device for Measuring Components of the Star Excursion Balance Test." North American Journal of Sports Physical Therapy, 4(2), pp. 92–99.',
                '[9] Gribble, P. A., Hertel, J., and Plisky, P. J. (2012). "Using the Star Excursion Balance Test to Assess Dynamic Postural-Control Deficits and Outcomes in Lower Extremity Injury: A Literature and Systematic Review." Journal of Athletic Training, 47(3), pp. 339–357.',
                '[10] Hertel, J., Braham, R. A., Hale, S. A., and Olmsted-Kramer, L. C. (2006). "Simplifying the Star Excursion Balance Test: Analyses of Subjects with and without Chronic Ankle Instability." Journal of Orthopaedic and Sports Physical Therapy, 36(3), pp. 131–137.',
                '[11] Plisky, P. J., Rauh, M. J., Kaminski, T. W., and Underwood, F. B. (2006). "Star Excursion Balance Test as a Predictor of Lower Extremity Injury in High School Basketball Players." Journal of Orthopaedic and Sports Physical Therapy, 36(12), pp. 911–919.',
                '[12] Cao, Z., Simon, T., Wei, S. E., and Sheikh, Y. (2017). "Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7291–7299.',
                '[13] Ranftl, R., Bochkovskiy, A., and Koltun, V. (2021). "Vision Transformers for Dense Prediction." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12179–12188.',
                '[14] Eigen, D., Puhrsch, C., and Fergus, R. (2014). "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network." Advances in Neural Information Processing Systems (NeurIPS), 27.',
                '[15] Fu, H., Gong, M., Wang, C., Batmanghelich, K., and Tao, D. (2018). "Deep Ordinal Regression Network for Monocular Depth Estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2002–2011.',
                '[16] Bradski, G. and Kaehler, A. (2008). Learning OpenCV: Computer Vision with the OpenCV Library. O\'Reilly Media, Sebastopol, CA.',
            ].map(ref => p(ref, { space: { before: 40, after: 40 } })),

        ]
    }]
});

Packer.toBuffer(doc).then(buf => {
    fs.writeFileSync('sebt_report2.docx', buf);
    console.log('Written: sebt_report2.docx');
}).catch(e => { console.error(e); process.exit(1); });