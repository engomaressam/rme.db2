interface RoleCardProps {
  title: string;
  responsibilities: string[];
}

const RoleCard = ({ title, responsibilities }: RoleCardProps) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h3 className="text-xl font-semibold mb-4 text-blue-800">{title}</h3>
      <ul className="list-disc pl-5 space-y-2">
        {responsibilities.map((item, index) => (
          <li key={index} className="text-gray-700">{item}</li>
        ))}
      </ul>
    </div>
  );
};

export default RoleCard;
