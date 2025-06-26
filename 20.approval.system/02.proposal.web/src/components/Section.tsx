interface SectionProps {
  title: string;
  id: string;
  children: React.ReactNode;
  bgColor?: string;
}

const Section = ({ title, id, children, bgColor = "bg-white" }: SectionProps) => {
  return (
    <section id={id} className={`py-12 ${bgColor}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-8">{title}</h2>
        {children}
      </div>
    </section>
  );
};

export default Section;
