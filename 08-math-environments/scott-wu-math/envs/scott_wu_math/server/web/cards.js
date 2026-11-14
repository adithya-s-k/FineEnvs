// Playing cards drawn in HTML: corner indices, the standard pip layouts for A–10, framed court cards.
(function () {
  const SUIT = { S: "♠", H: "♥", D: "♦", C: "♣" };
  const SUIT_NAME = { S: "spades", H: "hearts", D: "diamonds", C: "clubs" };
  const NAME = { 1: "A", 11: "J", 12: "Q", 13: "K" };
  const RANK_NAME = { 1: "Ace", 11: "Jack", 12: "Queen", 13: "King" };

  // Pip positions as [column, row, upside-down]. Columns: 0 left, 1 centre, 2 right.
  // Rows are percentages of the pip area, the classic layout of a French-suited deck.
  const L = 0, C = 1, R = 2;
  const PIPS = {
    2: [[C, 0], [C, 100, 1]],
    3: [[C, 0], [C, 50], [C, 100, 1]],
    4: [[L, 0], [R, 0], [L, 100, 1], [R, 100, 1]],
    5: [[L, 0], [R, 0], [C, 50], [L, 100, 1], [R, 100, 1]],
    6: [[L, 0], [R, 0], [L, 50], [R, 50], [L, 100, 1], [R, 100, 1]],
    7: [[L, 0], [R, 0], [C, 25], [L, 50], [R, 50], [L, 100, 1], [R, 100, 1]],
    8: [[L, 0], [R, 0], [C, 25], [L, 50], [R, 50], [C, 75, 1], [L, 100, 1], [R, 100, 1]],
    9: [[L, 0], [R, 0], [L, 33.3], [R, 33.3], [C, 50], [L, 66.7, 1], [R, 66.7, 1], [L, 100, 1], [R, 100, 1]],
    10: [[L, 0], [R, 0], [C, 16.7], [L, 33.3], [R, 33.3], [L, 66.7, 1], [R, 66.7, 1], [C, 83.3, 1], [L, 100, 1], [R, 100, 1]],
  };

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function corner(name, suit, flip) {
    const c = el("span", "corner" + (flip ? " flip" : ""));
    c.append(el("span", "corner-rank", name), el("span", "corner-suit", suit));
    return c;
  }

  // A real card from the deck.
  function playingCard(card) {
    const name = NAME[card.rank] || String(card.rank);
    const suit = SUIT[card.suit];
    const node = el("div", "card " + (card.suit === "H" || card.suit === "D" ? "red" : "black"));
    node.setAttribute("aria-label", `${RANK_NAME[card.rank] || card.rank} of ${SUIT_NAME[card.suit]}, value ${card.rank}`);
    node.append(corner(name, suit, false), corner(name, suit, true));
    const face = el("div", "face");
    if (card.rank === 1) {
      face.classList.add("ace");
      face.append(el("span", "pip big", suit));
    } else if (card.rank > 10) {
      face.classList.add("court");
      const frame = el("div", "frame");
      frame.append(el("span", "court-suit", suit), el("span", "court-letter", name), el("span", "court-suit flip", suit));
      face.append(frame);
    } else {
      face.classList.add("pips");
      for (const [col, row, flip] of PIPS[card.rank]) {
        const p = el("span", "pip" + (flip ? " flip" : ""), suit);
        p.style.left = ["22%", "50%", "78%"][col];
        p.style.top = row + "%";
        face.append(p);
      }
    }
    node.append(face);
    return node;
  }

  // A pile made by combining cards: shows its value and how it was made.
  function resultCard(valueText, exprText, count) {
    const node = el("div", "card made");
    node.setAttribute("aria-label", `Combined pile of ${count} cards, value ${valueText}: ${exprText}`);
    node.append(el("span", "made-count", count + " cards"));
    const v = el("span", "made-value", valueText);
    if (valueText.length > 5) v.classList.add("long");
    if (valueText.length > 8) v.classList.add("longer");
    node.append(v, el("span", "made-expr", exprText));
    node.title = exprText + " = " + valueText;
    return node;
  }

  window.Cards = { playingCard, resultCard, SUIT, NAME };
})();
