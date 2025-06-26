interface FlowchartProps {
  title: string;
  description: string;
  imagePath: string;
}

const Flowchart = ({ title, description, imagePath }: FlowchartProps) => {
  return (
    <div className="mb-12 bg-gray-50 p-6 rounded-lg shadow-sm">
      <h3 className="text-xl font-semibold mb-4">{title}</h3>
      <p className="mb-6 text-gray-700">{description}</p>
      <div className="flex justify-center">
        <img 
          src={imagePath} 
          alt={`${title} flowchart`} 
          className="max-w-full h-auto border rounded-lg shadow-md"
        />
      </div>
    </div>
  );
};

export default Flowchart;
