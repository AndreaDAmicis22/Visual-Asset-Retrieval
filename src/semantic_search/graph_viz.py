"""
graph_viz.py — Visualizzazione interattiva del grafo con Pyvis
==============================================================
Genera file HTML apribili nel browser con nodi trascinabili,
zoom e click per dettagli.

Funzioni:
  - viz_subgraph   : sottografo attorno a un'immagine (vicini + entità)
        poetry run python main.py --viz-subgraph 000000440507.jpg --graph-depth 2
  - viz_clusters   : cluster di immagini simili (community detection)
        poetry run python main.py --viz-clusters
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from semantic_search.config import DEFAULT_MODEL, PROJECT_ROOT
from semantic_search.image_server import BASE_URL, start_server

OUTPUT_DIR = PROJECT_ROOT / "data" / "graphs"

SIMILAR_TO = "SIMILAR_TO"
CONTAINS = "CONTAINS"

# ── Colori e stili ────────────────────────────────────
COLOR_IMAGE = "#4f86f7"  # blu     — nodi immagine
COLOR_ENTITY = "#f7a24f"  # arancio — nodi entità
COLOR_EDGE_SIM = "#4ade80"  # verde   — archi SIMILAR_TO
COLOR_EDGE_CON = "#94a3b8"  # grigio  — archi CONTAINS
COLOR_CLUSTER = [  # palette cluster
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#e91e63",
    "#00bcd4",
    "#8bc34a",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
# Pannello laterale con foto — iniettato nell'HTML di Pyvis dopo la generazione
_IMAGE_PANEL_JS = """
<style>
  #img-panel {{
    position: fixed;
    top: 20px;
    right: 20px;
    width: 380px;
    background: #1e1e2e;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 12px;
    display: none;
    z-index: 9999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    font-family: sans-serif;
    color: white;
  }}
  #img-panel img {{
    width: 100%;
    border-radius: 6px;
    margin-bottom: 8px;
    max-height: 400px;
    object-fit: cover;
    }}
  #img-panel-name {{
    font-size: 13px;
    font-weight: bold;
    word-break: break-all;
    margin-bottom: 4px;
  }}
  #img-panel-caption {{
    font-size: 12px;
    color: #aaa;
    line-height: 1.4;
  }}
  #img-panel-close {{
    position: absolute;
    top: 8px;
    right: 10px;
    cursor: pointer;
    font-size: 18px;
    color: #fff;
    text-shadow: 0 0 4px rgba(0, 0, 0, 0.8);
    z-index: 10;
  }}
  #img-panel-close:hover {{ color: white; }}
</style>

<div id="img-panel">
  <span id="img-panel-close" onclick="document.getElementById('img-panel').style.display='none'">✕</span>
  <img id="img-panel-img" src="" alt="">
  <div id="img-panel-name"></div>
  <div id="img-panel-caption"></div>
</div>

<script>
// Metadati immagini iniettati da Python
const IMAGE_META = {image_meta_json};
const BASE_URL   = "{base_url}";

// Intercetta click sui nodi dopo che Pyvis ha inizializzato la rete
window.addEventListener("load", function() {{
  // Pyvis espone la variabile `network` globalmente
  function hookNetwork() {{
    if (typeof network === "undefined") {{
      setTimeout(hookNetwork, 200);
      return;
    }}
    network.on("click", function(params) {{
      if (params.nodes.length === 0) return;

      const nodeId  = params.nodes[0];
      const meta    = IMAGE_META[nodeId];
      if (!meta) return;   // nodo entità — ignora

      const panel   = document.getElementById("img-panel");
      const img     = document.getElementById("img-panel-img");
      const name    = document.getElementById("img-panel-name");
      const caption = document.getElementById("img-panel-caption");

      img.src     = BASE_URL + "/images/" + nodeId;
      name.textContent    = nodeId;
      caption.innerHTML = "<ul style='margin:4px 0 0 16px; padding:0;'>" +
        (meta.captions || [])
          .map((c, i) => `<li style="opacity:${{1 - i * 0.15}}; margin-bottom:3px;">${{c}}</li>`)
          .join("") +
        "</ul>";
      panel.style.display = "block";
    }});
  }}
  hookNetwork();
}});
</script>
"""


def _inject_panel(html_path: Path, G: nx.DiGraph):
    """
    Inietta il pannello laterale con JS nell'HTML generato da Pyvis.
    Viene chiamato dopo save_graph().
    """

    image_meta = {}
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "image":
            caps = data.get("captions", [])
            # image_meta[node] = {"caption": caps[0] if caps else ""}
            image_meta[node] = {"captions": caps}

    panel_html = _IMAGE_PANEL_JS.format(
        image_meta_json=json.dumps(image_meta, ensure_ascii=False),
        base_url=BASE_URL,
    )

    content = html_path.read_text(encoding="utf-8")
    content = content.replace("</body>", panel_html + "\n</body>")
    html_path.write_text(content, encoding="utf-8")


def _image_url(filename: str) -> str:
    """Restituisce l'URL HTTP dell'immagine servita dal server locale."""
    return f"{BASE_URL}/images/{filename}"


def _image_tooltip(node: str, data: dict) -> str:
    """Tooltip semplice per hover — solo nome e prima didascalia."""
    caps = data.get("captions", [])
    return caps[0] if caps else node


def _entity_tooltip(node: str) -> str:
    """Tooltip semplice per hover — solo il nome dell'entità."""
    return node


def _base_network(height: str = "100vh") -> Network:
    """Crea una Network Pyvis con impostazioni di base."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
    )
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 120,
          "springConstant": 0.04
        },
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.1 },
        "font": { "size": 7, "align": "middle" }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)
    return net


def _save_and_open(net: Network, out: Path, open_browser: bool, G: nx.DiGraph) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out))
    _inject_panel(out, G)  # ← inietta pannello
    print(f"  Grafo salvato: {out}")
    if open_browser:
        webbrowser.open(out.as_uri())
    return out


# ─────────────────────────────────────────────
# viz_subgraph
# ─────────────────────────────────────────────


def viz_subgraph(
    G: nx.DiGraph,
    filename: str,
    depth: int = 1,
    model_name: str = DEFAULT_MODEL,
    open_browser: bool = True,
) -> Path:
    """
    Visualizza il sottografo attorno a `filename` entro `depth` hop.

    Nodi gialli  = immagine centrale
    Nodi blu     = immagini vicine
    Nodi arancio = entità estratte dalle didascalie
    Archi verdi  = SIMILAR_TO (spessore proporzionale allo score)
    Archi grigi  = CONTAINS

    Clicca su un nodo immagine per vedere la foto nel tooltip.
    """
    start_server()
    if filename not in G:
        msg = f"Immagine non trovata nel grafo: {filename}"
        raise KeyError(msg)

    # Estrai sottografo
    nodes_in_range = nx.single_source_shortest_path_length(G, filename, cutoff=depth)
    sub = G.subgraph(nodes_in_range.keys()).copy()

    net = _base_network()

    # ── Nodi ──────────────────────────────────────────
    for node, data in sub.nodes(data=True):
        is_image = data.get("node_type") == "image"
        is_center = node == filename

        if is_image:
            tooltip = _image_tooltip(node, data)
            color = "#f1c40f" if is_center else COLOR_IMAGE
            size = 30 if is_center else 18
            shape = "box"
            border = 3 if is_center else 1
        else:
            tooltip = _entity_tooltip(node)
            color = COLOR_ENTITY
            size = 10
            shape = "ellipse"
            border = 1

        net.add_node(
            node,
            label=node,
            title=tooltip,
            color=color,
            size=size,
            shape=shape,
            borderWidth=border,
        )

    # ── Archi ──────────────────────────────────────────
    for src, dst, data in sub.edges(data=True):
        relation = data.get("relation")
        if relation == SIMILAR_TO:
            w = data.get("weight", 0)
            net.add_edge(
                src,
                dst,
                title=f"score: {w:.3f}",
                label=f"SIMILAR TO {w:.2f}",
                color=COLOR_EDGE_SIM,
                width=0.5 + w * 2,
                font={"size": 7, "color": "#8df0b1"},
            )
        elif relation == CONTAINS:
            net.add_edge(
                src,
                dst,
                label="CONTAINS",
                color=COLOR_EDGE_CON,
                width=0.5,
                arrows="to",
                font={"size": 7, "color": "#94a3b8"},
            )

    stem = Path(filename).stem
    out = OUTPUT_DIR / f"subgraph_{stem}_depth{depth}.html"
    return _save_and_open(net, out, open_browser, sub)


# ─────────────────────────────────────────────
# viz_clusters
# ─────────────────────────────────────────────
def viz_clusters(
    G: nx.DiGraph,
    model_name: str = DEFAULT_MODEL,
    max_nodes: int = 300,
    open_browser: bool = True,
) -> Path:
    """
    Visualizza i cluster di immagini simili usando community detection
    (algoritmo di Louvain su grafo non diretto delle similarità).

    Ogni cluster ha un colore diverso.
    Il nodo rappresentante (stella) è il più connesso del cluster —
    cliccaci sopra per vedere la foto associata nel pannello laterale.
    Le entità non vengono mostrate per mantenere il grafo leggibile.
    """
    start_server()

    # ── Costruisce grafo solo Image + SIMILAR_TO ───────
    image_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "image"]
    image_set = set(image_nodes)

    G_images = nx.DiGraph()
    for src, dst, data in G.edges(data=True):
        if data.get("relation") == SIMILAR_TO and src in image_set and dst in image_set:
            G_images.add_edge(src, dst, weight=data.get("weight", 0))

    # ── Limita i nodi per leggibilità ─────────────────
    if G_images.number_of_nodes() > max_nodes:
        print(f"  [INFO] Grafo ridotto a {max_nodes} nodi (totale: {G_images.number_of_nodes()})")
        top_nodes = sorted(G_images.nodes(), key=G_images.degree, reverse=True)[:max_nodes]
        G_images = G_images.subgraph(top_nodes).copy()

    # ── Community detection — Louvain ─────────────────
    G_undirected = G_images.to_undirected()
    node_to_cluster: dict[str, int] = {}
    representatives: set[str] = set()

    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(G_undirected, seed=42)
        for cluster_id, community in enumerate(communities):
            for node in community:
                node_to_cluster[node] = cluster_id
            # Rappresentante = nodo più connesso del cluster
            rep = max(community, key=G_images.degree)
            representatives.add(rep)

        print(f"  Community rilevate: {len(communities)}")

    except Exception:
        node_to_cluster = dict.fromkeys(G_images.nodes(), 0)
        print("  [WARN] Louvain non disponibile — tutti i nodi nello stesso cluster")

    net = _base_network(height="100vh")

    # ── Nodi ──────────────────────────────────────────
    # I rappresentanti vengono stilati come stelle grandi,
    # ma mantengono il loro filename come ID — così IMAGE_META
    # li trova e il click apre la foto nel pannello laterale.
    for node in G_images.nodes():
        cluster_id = node_to_cluster.get(node, 0)
        color = COLOR_CLUSTER[cluster_id % len(COLOR_CLUSTER)]
        node_data = G.nodes[node]
        tooltip = _image_tooltip(node, node_data)
        is_rep = node in representatives

        if is_rep:
            net.add_node(
                node,
                label=f"Cluster {cluster_id}",
                title=tooltip,
                color=color,
                size=45,
                shape="circularImage",
                image=_image_url(node),
                borderWidth=3,
                font={"size": 12, "bold": True},
            )
        else:
            net.add_node(
                node,
                label="",
                title=tooltip,
                color=color,
                size=14,
                shape="dot",
                borderWidth=1,
                font={"size": 0},
            )

    # ── Archi ──────────────────────────────────────────
    for src, dst, data in G_images.edges(data=True):
        w = data.get("weight", 0)
        net.add_edge(
            src,
            dst,
            title=f"score: {w:.3f}",
            label=f"{w:.2f}",
            color={"color": "#ffffff", "opacity": 0.15},
            width=0.5 + w * 2,
            font={"size": 7, "color": "#ffffff", "strokeWidth": 0, "background": "none"},
        )

    out = OUTPUT_DIR / "clusters.html"
    return _save_and_open(net, out, open_browser, G)
