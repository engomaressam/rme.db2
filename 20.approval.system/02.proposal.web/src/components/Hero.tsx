const Hero = () => {
  return (
    <div className="bg-gradient-to-r from-blue-800 to-blue-600 text-white py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl md:text-6xl">
            Construction Document Approval Tracking System
          </h1>
          <p className="mt-6 max-w-3xl mx-auto text-xl">
            A dedicated web-based system designed to streamline and manage crucial document approval processes 
            between Contractors, Consultants, and Owners on construction projects.
          </p>
          <div className="mt-10">
            <a
              href="#features"
              className="inline-block bg-white text-blue-800 px-8 py-3 rounded-md text-base font-medium shadow hover:bg-gray-100"
            >
              Explore Features
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
