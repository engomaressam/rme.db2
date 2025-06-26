import { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

interface MermaidDiagramProps {
  chart: string;
  id: string;
}

const MermaidDiagram = ({ chart, id }: MermaidDiagramProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      flowchart: {
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'basis'
      },
      securityLevel: 'loose'
    });
    
    if (containerRef.current) {
      mermaid.render(`mermaid-${id}`, chart).then(({ svg }) => {
        if (containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      });
    }
  }, [chart, id]);

  return (
    <div className="mermaid-container bg-gray-50 p-6 rounded-lg shadow-sm mb-12">
      <div ref={containerRef} className="flex justify-center"></div>
    </div>
  );
};

export default MermaidDiagram;
