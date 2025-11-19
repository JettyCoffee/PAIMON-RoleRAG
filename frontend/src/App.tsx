import React, { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Send, User, Bot, Network } from 'lucide-react';

interface Node {
    id: string;
    type?: string;
    val?: number;
    [key: string]: any;
}

interface Link {
    source: string;
    target: string;
    description?: string;
    [key: string]: any;
}

interface GraphData {
    nodes: Node[];
    links: Link[];
}

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

function App() {
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const fgRef = useRef<any>();

    useEffect(() => {
        // Fetch Graph Data
        fetch('/api/graph')
            .then(res => res.json())
            .then(data => {
                // Transform NetworkX JSON to react-force-graph format if needed
                // NetworkX node-link-data usually has "nodes" and "links"
                if (data.nodes) {
                    setGraphData(data);
                }
            })
            .catch(err => console.error("Failed to fetch graph:", err));
    }, []);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = input;
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setInput('');
        setLoading(true);

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg, role: "Harry Potter" })
            });
            const data = await res.json();
            setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
        } catch (err) {
            console.error("Chat failed:", err);
            setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not reach the wizard." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex h-screen bg-slate-900 text-slate-100 overflow-hidden">
            {/* Left Panel: Graph */}
            <div className="w-2/3 h-full border-r border-slate-700 relative">
                <div className="absolute top-4 left-4 z-10 bg-slate-800/80 p-2 rounded backdrop-blur">
                    <h2 className="flex items-center gap-2 font-bold text-purple-400">
                        <Network size={20} /> Knowledge Graph
                    </h2>
                    <p className="text-xs text-slate-400">Nodes: {graphData.nodes.length} | Links: {graphData.links.length}</p>
                </div>
                <ForceGraph2D
                    ref={fgRef}
                    graphData={graphData}
                    nodeLabel="id"
                    nodeAutoColorBy="type"
                    linkDirectionalArrowLength={3.5}
                    linkDirectionalArrowRelPos={1}
                    backgroundColor="#0f172a"
                    nodeCanvasObject={(node, ctx, globalScale) => {
                        const label = node.id;
                        const fontSize = 12 / globalScale;
                        ctx.font = `${fontSize}px Sans-Serif`;
                        const textWidth = ctx.measureText(label).width;
                        const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

                        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                        if (node.type === 'character') ctx.fillStyle = 'rgba(147, 51, 234, 0.2)'; // Purple for chars

                        ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);

                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillStyle = node.color || '#cbd5e1';
                        ctx.fillText(label, node.x, node.y);

                        node.__bckgDimensions = bckgDimensions; // to re-use in nodePointerAreaPaint
                    }}
                />
            </div>

            {/* Right Panel: Chat */}
            <div className="w-1/3 h-full flex flex-col bg-slate-800">
                <div className="p-4 border-b border-slate-700 bg-slate-900/50">
                    <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                        PAIMON: RoleRAG
                    </h1>
                    <p className="text-sm text-slate-400">Chatting with Harry Potter</p>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-blue-600' : 'bg-purple-600'
                                }`}>
                                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                            </div>
                            <div className={`max-w-[80%] p-3 rounded-lg text-sm ${msg.role === 'user' ? 'bg-blue-600/20 border border-blue-500/30' : 'bg-purple-600/20 border border-purple-500/30'
                                }`}>
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center animate-pulse">
                                <Bot size={16} />
                            </div>
                            <div className="bg-purple-600/20 p-3 rounded-lg text-sm text-slate-400">
                                Thinking...
                            </div>
                        </div>
                    )}
                </div>

                <div className="p-4 border-t border-slate-700 bg-slate-900/50">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="Ask something..."
                            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 focus:outline-none focus:border-purple-500 transition-colors"
                        />
                        <button
                            onClick={handleSend}
                            disabled={loading}
                            className="bg-purple-600 hover:bg-purple-500 text-white p-2 rounded-lg transition-colors disabled:opacity-50"
                        >
                            <Send size={20} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
