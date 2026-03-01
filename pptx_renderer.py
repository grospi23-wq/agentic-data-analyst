from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

# Need to import the Pydantic models to use as type hints
from schema import PresentationSpec, SlideContent

THEME = {
    "bg": RGBColor(0x0D, 0x1B, 0x2A),
    "accent": RGBColor(0x00, 0xB4, 0xD8),
    "text": RGBColor(0xFF, 0xFF, 0xFF),
    "subtext": RGBColor(0xB0, 0xC4, 0xDE),
    "warn": RGBColor(0xFF, 0xA5, 0x00),
    "good": RGBColor(0x00, 0xFF, 0x00),
    "bad": RGBColor(0xFF, 0x00, 0x00)
}

def _add_slide(prs: Presentation, layout_index: int) -> any:
    layout = prs.slide_layouts[layout_index]
    return prs.slides.add_slide(layout)

def _style_textframe(tf, color: RGBColor, size_pt: int, bold: bool = False, align=PP_ALIGN.LEFT):
    for para in tf.paragraphs:
        para.alignment = align
        for run in para.runs:
            run.font.color.rgb = color
            run.font.size = Pt(size_pt)
            run.font.bold = bold

def _render_title_slide(slide, spec: SlideContent, full_spec: PresentationSpec, prs: Presentation):
    title_box = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11.33), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = full_spec.deck_title
    _style_textframe(tf, THEME["accent"], 44, bold=True, align=PP_ALIGN.CENTER)

    sub_box = slide.shapes.add_textbox(Inches(1), Inches(4), Inches(11.33), Inches(1))
    tf_sub = sub_box.text_frame
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = full_spec.subtitle
    _style_textframe(tf_sub, THEME["subtext"], 24, align=PP_ALIGN.CENTER)

def _render_hypothesis_slide(slide, spec: SlideContent):
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Hypothesis & Verdict"
    _style_textframe(tf, THEME["accent"], 32, bold=True)

    body_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(12), Inches(4))
    tf_b = body_box.text_frame
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.text = spec.title # Assuming title contains the verdict text
    _style_textframe(tf_b, THEME["text"], 20)

def _render_finding_slide(slide, spec: SlideContent):
    # Title bar (left accent stripe)
    accent_bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0), Inches(0.12), Inches(7.5)
    )
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = THEME["accent"]
    accent_bar.line.fill.background()

    # Slide title
    title_box = slide.shapes.add_textbox(Inches(0.3), Inches(0.3), Inches(9), Inches(0.9))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = spec.title
    p.runs[0].font.color.rgb = THEME["accent"]
    p.runs[0].font.size = Pt(24)
    p.runs[0].font.bold = True

    # Metric callout (right side, large)
    if spec.metric_callout:
        callout = slide.shapes.add_textbox(Inches(10), Inches(1.5), Inches(3), Inches(2))
        tf_c = callout.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = spec.metric_callout
        p_c.alignment = PP_ALIGN.CENTER
        p_c.runs[0].font.color.rgb = THEME["accent"]
        p_c.runs[0].font.size = Pt(36)
        p_c.runs[0].font.bold = True

    # Bullet body
    body_box = slide.shapes.add_textbox(Inches(0.3), Inches(1.4), Inches(9.5), Inches(5.5))
    tf_b = body_box.text_frame
    tf_b.word_wrap = True
    for i, bullet in enumerate(spec.body_bullets):
        p_b = tf_b.paragraphs[0] if i == 0 else tf_b.add_paragraph()
        p_b.text = f"• {bullet}"
        p_b.runs[0].font.color.rgb = THEME["subtext"]
        p_b.runs[0].font.size = Pt(16)
        p_b.space_after = Pt(8)

def _render_limitation_slide(slide, spec: SlideContent):
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "⚠️ Data Limitations"
    _style_textframe(tf, THEME["warn"], 32, bold=True)

    body_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(12), Inches(4))
    tf_b = body_box.text_frame
    tf_b.word_wrap = True
    for i, bullet in enumerate(spec.body_bullets):
        p_b = tf_b.paragraphs[0] if i == 0 else tf_b.add_paragraph()
        p_b.text = f"• {bullet}"
        p_b.runs[0].font.color.rgb = THEME["text"]
        p_b.runs[0].font.size = Pt(18)
        p_b.space_after = Pt(12)

def _render_conclusion_slide(slide, spec: SlideContent, critic_score: float):
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Conclusion"
    _style_textframe(tf, THEME["accent"], 32, bold=True)
    
    body_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(8), Inches(4))
    tf_b = body_box.text_frame
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.text = spec.title
    _style_textframe(tf_b, THEME["text"], 20)

    # Score Box
    score_color = THEME["good"] if critic_score >= 0.8 else THEME["warn"]
    score_box = slide.shapes.add_textbox(Inches(9), Inches(2), Inches(3), Inches(2))
    tf_s = score_box.text_frame
    p_s = tf_s.paragraphs[0]
    p_s.text = f"Reliability Score\n{critic_score:.2f}/1.0"
    _style_textframe(tf_s, score_color, 24, bold=True, align=PP_ALIGN.CENTER)

def render_pptx(spec: PresentationSpec, output_path: str) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.33)   # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    LAYOUT_BLANK = 6  # Blank layout gives us full control

    for slide_spec in spec.slides:
        slide = _add_slide(prs, LAYOUT_BLANK)

        # --- Background fill ---
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = THEME["bg"]

        if slide_spec.slide_type == "title":
            _render_title_slide(slide, slide_spec, spec, prs)
        elif slide_spec.slide_type == "hypothesis":
            _render_hypothesis_slide(slide, slide_spec)
        elif slide_spec.slide_type == "finding":
            _render_finding_slide(slide, slide_spec)
        elif slide_spec.slide_type == "limitation":
            _render_limitation_slide(slide, slide_spec)
        elif slide_spec.slide_type == "conclusion":
            _render_conclusion_slide(slide, slide_spec, spec.critic_score)

        # Speaker notes
        if slide_spec.speaker_notes:
            slide.notes_slide.notes_text_frame.text = slide_spec.speaker_notes

    out = Path(output_path)
    prs.save(out)
    return out