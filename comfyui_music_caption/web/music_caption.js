import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Show the formatted report right inside `MusicCaptionView` when a run finishes
// (the shipped workflow also wires `PreviewAny` nodes so text displays on both
// classic LiteGraph and Nodes 2.0 frontends).
app.registerExtension({
    name: "audio_gen_continue.music_caption_view",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "MusicCaptionView") {
            return;
        }
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            const lines = Array.isArray(message?.text) ? message.text : [];
            const content = lines.join("\n").trim();
            if (!content) {
                return;
            }
            let preview = this.widgets?.find((w) => w.name === "_caption_preview");
            if (!preview) {
                const created = ComfyWidgets["STRING"](
                    this,
                    "_caption_preview",
                    ["STRING", { multiline: true }],
                    appInstance
                );
                preview = created.widget;
                preview.serialize = false;
                if (preview.inputEl) {
                    preview.inputEl.readOnly = true;
                    preview.inputEl.style.opacity = "0.92";
                    preview.inputEl.style.fontFamily = "monospace";
                    preview.inputEl.style.fontSize = "12px";
                }
            }
            preview.value = content;
            this.onResize?.(this.size);
            appInstance.graph?.setDirtyCanvas(true, true);
        };
    },
});
