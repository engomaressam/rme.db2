import { useState } from 'react';
import { Menu, X } from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  return (
    <nav className="bg-blue-800 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <span className="font-bold text-xl">Construction Approval Hub</span>
            </div>
          </div>
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <a href="#executive-summary" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
                Overview
              </a>
              <a href="#features" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
                Features
              </a>
              <a href="#workflows" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
                Workflows
              </a>
              <a href="#benefits" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
                Benefits
              </a>
              <a href="#next-steps" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
                Next Steps
              </a>
            </div>
          </div>
          <div className="md:hidden">
            <button
              onClick={toggleMenu}
              className="inline-flex items-center justify-center p-2 rounded-md text-white hover:bg-blue-700 focus:outline-none"
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            <a
              href="#executive-summary"
              className="block px-3 py-2 rounded-md text-base font-medium hover:bg-blue-700"
              onClick={toggleMenu}
            >
              Overview
            </a>
            <a
              href="#features"
              className="block px-3 py-2 rounded-md text-base font-medium hover:bg-blue-700"
              onClick={toggleMenu}
            >
              Features
            </a>
            <a
              href="#workflows"
              className="block px-3 py-2 rounded-md text-base font-medium hover:bg-blue-700"
              onClick={toggleMenu}
            >
              Workflows
            </a>
            <a
              href="#benefits"
              className="block px-3 py-2 rounded-md text-base font-medium hover:bg-blue-700"
              onClick={toggleMenu}
            >
              Benefits
            </a>
            <a
              href="#next-steps"
              className="block px-3 py-2 rounded-md text-base font-medium hover:bg-blue-700"
              onClick={toggleMenu}
            >
              Next Steps
            </a>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
