import { motion } from "framer-motion";
import { useTheme } from "../../ThemeContext";
import { MONO } from "../../theme";
import { Accent } from "../../components/primitives";
import { HFMark } from "../../components/figures";

export function DemoIntroSlide() {
  const { T } = useTheme();
  const fade = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0 } };
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <motion.div
        initial="hidden"
        animate="show"
        variants={{ show: { transition: { staggerChildren: 0.24, delayChildren: 0.2 } } }}
        style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: "0 120px" }}
      >
        <motion.div variants={fade} transition={{ type: "spring", damping: 20 }} style={{ fontFamily: MONO, fontSize: 22, letterSpacing: 6, color: T.textDim, textTransform: "uppercase", marginBottom: 30 }}>
          Hands-on
        </motion.div>
        <motion.div variants={fade} transition={{ type: "spring", damping: 20 }} style={{ fontSize: 62, fontWeight: 800, color: T.white, lineHeight: 1.15 }}>
          Let’s <Accent color="emerald">train a model</Accent>.
        </motion.div>
        <motion.div variants={fade} transition={{ type: "spring", damping: 20 }} style={{ marginTop: 28, fontSize: 30, color: T.textMuted, maxWidth: 980, lineHeight: 1.4 }}>
          We’ll build the <b style={{ color: T.white }}>latex-ocr-env</b> environment and train a VLM
          against it — with <Accent color="emerald">TRL + GRPO</Accent>.
        </motion.div>
        <motion.div variants={fade} transition={{ type: "spring", damping: 20 }} style={{ marginTop: 36, display: "flex", alignItems: "center", gap: 12, fontFamily: MONO, fontSize: 20, color: T.textDim }}>
          <HFMark size={26} />
          <span>hf.co/spaces/HuggingEnvs/latex-ocr-env</span>
        </motion.div>
      </motion.div>
    </div>
  );
}
