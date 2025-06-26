import React, { useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Section from './components/Section';
import MermaidDiagram from './components/MermaidDiagram';
import FeatureCard from './components/FeatureCard';
import RoleCard from './components/RoleCard';
import './index.css';

function App() {
  useEffect(() => {
    // Scroll to hash on load if present
    if (window.location.hash) {
      const id = window.location.hash.substring(1);
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }, []);

  // Mermaid diagram definitions
  const irFlowchart = `
    flowchart TD
      %% Define nodes
      subgraph Contractor
        A[Select activity from<br>predefined list]
        B[Submit IR<br>Rev.00<br>with attachments]
      end
      
      subgraph Consultant
        C[Review IR]
        D[Code A, B, C, or D?]
        E[Add notes]
      end
      
      subgraph Owner
        F[View access]
        G[End]
        H[Resubmit IR<br>Rev.01, Rev.02, etc.]
        I[Work on site]
      end
      
      %% Define connections
      A --> B
      B --> C
      C --> D
      D -->|A, B, D| G
      D -->|C| E
      E --> H
      H --> I
      I --> C
  `;

  const sdFlowchart = `
    flowchart TD
      %% Define nodes
      subgraph Contractor
        A[Submit SD<br>PDF/DWG]
        B[Revise SD]
        C[Resubmit SD<br>new revision]
      end
      
      subgraph Consultant
        D[Review SD]
        E[Code A, B, or C?]
        F[Add notes]
        G[Receive notification]
      end
      
      subgraph Owner
        H[View access]
        I[End]
      end
      
      %% Define connections
      A --> D
      D --> E
      E -->|A| I
      E -->|B, C| F
      F --> G
      G --> B
      B --> C
      C --> D
  `;

  const materialFlowchart = `
    flowchart TD
      %% Define nodes
      subgraph Contractor
        A[Material Submittal<br>specs/attachments]
        B[Receive notification]
        C[Revise submittal]
      end
      
      subgraph Consultant
        D[Review submittal]
        E[Code A, B, or C?]
        F[Add notes]
        G[Material Submittal<br>new revision]
      end
      
      subgraph Owner
        H[View access]
        I[End]
      end
      
      %% Define connections
      A --> D
      D --> E
      E -->|A| I
      E -->|B, C| F
      F --> B
      B --> C
      C --> G
      G --> D
  `;

  return (
    <div className="App">
      <Navbar />
      <Hero />
      
      {/* Introduction */}
      <Section title="Introduction" id="introduction" bgColor="bg-white">
        <>
          <p className="text-lg mb-6">
            The Construction Document Approval Tracking System is a comprehensive web-based solution designed to streamline and automate the approval processes for construction projects. This updated proposal incorporates feedback from the consultant, ensuring the system meets all operational requirements, including discipline-based serial numbers for Inspection Requests, attachment support for all requests and responses, multi-stage approval workflows, and specialized handling for Pouring Requests and Code B closures.
          </p>
          <p className="text-lg mb-6">
            Our system addresses the critical need for organized tracking of inspection requests, shop drawings, material approvals, and other essential construction documents. By digitizing these workflows and supporting attachments at every stage, we eliminate the inefficiencies of paper-based processes while providing real-time visibility into approval statuses for all stakeholders.
          </p>
        </>
      </Section>

      {/* System Overview */}
      <Section title="System Overview" id="overview" bgColor="bg-gray-50">
        <>
          <p className="text-lg mb-8">
            The Construction Document Approval Tracking System is built around ten key document types, each with specific approval workflows:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <FeatureCard 
              title="Inspection Requests (IRs)" 
              description="Contractor-initiated requests for inspection of completed construction activities, selected from a predefined list. Each discipline (Civil, Mechanical, Electrical, etc.) maintains a separate serial number for proper tracking. Consultants review and assign codes (A/B/C/D). Attachments are supported for both submission and response."/>
            <FeatureCard 
              title="Shop Drawings (SDs)" 
              description="Technical drawings submitted by contractors for approval before fabrication. Consultants review and assign codes (A/B/C). Submissions may be closed with Code B if comments are minor; Code A is not mandatory for closure. Attachments are supported for all submissions and responses."/>
            <FeatureCard 
              title="Material Submittals" 
              description="Specifications and samples of materials proposed for use in construction. Follows similar approval workflow to shop drawings, including closure with Code B for minor comments. Attachments are supported for all submissions and responses."/>
            <FeatureCard 
              title="Pouring Requests" 
              description="Requests submitted by the contractor to the consultant for review and approval prior to pouring any structural element. Each request requires approval from Civil, Mechanical, and Electrical departments before proceeding."/>
            <FeatureCard 
              title="Non-Conformance Reports (NCRs)" 
              description="Documentation of work that doesn't meet specifications. Remains open until contractor resolves the issue. Attachments are supported for all submissions and responses."/>
            <FeatureCard 
              title="Material Inspection Requests (MIRs)" 
              description="Similar to IRs but specifically for inspecting delivered materials on site. Attachments are supported for all submissions and responses."/>
            <FeatureCard 
              title="Invoices" 
              description="Monthly contractor submissions that require consultant approval before being sent to the owner/client. Attachments are supported for all submissions and responses."/>
            <FeatureCard 
              title="Requests for Information (RFIs)" 
              description="Contractor questions requiring consultant clarification. Attachments are supported for all submissions and responses."/>
            <FeatureCard 
              title="Requests for Change (RFCs)" 
              description="Formal requests for changes to project scope, specifications, or timeline. Submitted by contractor, reviewed by consultant, and require final approval from the owner. Attachments are supported for all submissions and responses."/>
            <FeatureCard 
              title="Time Schedules" 
              description="Project timelines submitted by contractors for consultant review and approval, with final approval required from the owner. Attachments are supported for all submissions and responses."/>
          </div>
        </>
      </Section>

      {/* User Roles */}
      <Section title="User Roles and Permissions" id="roles" bgColor="bg-white">
        <>
          <p className="text-lg mb-8">
            The system supports three primary user roles, each with specific permissions and capabilities. All users can attach documents to requests and responses. Certain workflows, such as RFCs, Time Schedules, and Pouring Requests, require multi-stage or multi-department approval as described below:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <RoleCard 
              title="Contractor" 
              responsibilities={[
                "Submit inspection requests from predefined activity list (with discipline-based serials)",
                "Request additions to the activity list",
                "Submit shop drawings and material specifications (with attachments)",
                "Submit Pouring Requests requiring approval from Civil, Mechanical, and Electrical departments",
                "Respond to NCRs and address issues",
                "Submit invoices, RFIs, RFCs, and time schedules (with attachments)",
                "Track status of all submissions"
              ]}
            />
            <RoleCard 
              title="Consultant" 
              responsibilities={[
                "Define and manage activity lists",
                "Review and code all submissions (A/B/C/D)",
                "Add notes to submissions requiring revision",
                "Initiate NCRs for non-conforming work",
                "Approve or reject invoices",
                "Respond to RFIs and review RFCs and Time Schedules, forwarding to owner for final approval as needed",
                "Review Pouring Requests and coordinate multi-department approval",
                "Generate reports and monitor project status"
              ]}
            />
            <RoleCard 
              title="Owner/Client" 
              responsibilities={[
                "View all document statuses and approval histories",
                "Access approved invoices",
                "Provide final approval for RFCs and Time Schedules",
                "Monitor overall project progress",
                "View reports and analytics",
                "Maintain oversight without direct approval responsibilities except where specified"
              ]}
            />
          </div>
        </>
      </Section>

      {/* Approval Cycles */}
      <Section title="Approval Cycle Workflows" id="workflows" bgColor="bg-gray-50">
        <>
          <p className="text-lg mb-8">
            To visually represent the core approval workflows, we have developed flowcharts for the primary document types. These illustrate the interaction between roles and the decision points within the system:
          </p>
          <h3 className="text-xl font-semibold mb-4">Inspection Request (IR) Flow</h3>
          <p className="mb-6 text-gray-700">This chart shows the process from activity selection and submission by the Contractor, through Consultant review and coding (A/B/C/D), including the revision loop for rejected IRs. Each discipline maintains a separate serial number for tracking.</p>
          <MermaidDiagram chart={irFlowchart} id="ir-flow" />
          <h3 className="text-xl font-semibold mb-4">Shop Drawing (SD) Flow</h3>
          <p className="mb-6 text-gray-700">This chart details the submission, Consultant review (Codes A/B/C), and the revision cycle required for B/C coded submittals until closure. Submissions may be closed with Code B if comments are minor; Code A is not mandatory for closure.</p>
          <MermaidDiagram chart={sdFlowchart} id="sd-flow" />
          <h3 className="text-xl font-semibold mb-4">Material Submittal Flow</h3>
          <p className="mb-6 text-gray-700">This chart mirrors the SD flow but is specific to material specifications, showing the path to closure with Code B for minor comments. Attachments are supported for all submissions and responses.</p>
          <MermaidDiagram chart={materialFlowchart} id="material-flow" />
          <h3 className="text-xl font-semibold mb-4">Pouring Request Flow</h3>
          <p className="mb-6 text-gray-700">Pouring Requests are submitted by the contractor and require approval from Civil, Mechanical, and Electrical departments before pouring any structural element. All approvals must be obtained in the same request.</p>
          {/* Add a new MermaidDiagram for Pouring Requests if available */}
        </>
      </Section>

      {/* Benefits */}
      <Section title="Benefits of the Proposed System" id="benefits" bgColor="bg-white">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard 
            title="Increased Efficiency" 
            description="Eliminates paper-based processes and reduces approval cycle times by 40-60%. Automated notifications ensure timely responses."
          />
          <FeatureCard 
            title="Enhanced Transparency" 
            description="All stakeholders have real-time visibility into document status, approval history, and outstanding items."
          />
          <FeatureCard 
            title="Improved Accountability" 
            description="Clear tracking of responsibilities, timestamps on all actions, and complete audit trails for all documents."
          />
          <FeatureCard 
            title="Reduced Errors" 
            description="Structured workflows prevent common mistakes and ensure all required information is captured."
          />
          <FeatureCard 
            title="Better Communication" 
            description="Centralized platform for all project documentation with integrated commenting and notification systems."
          />
          <FeatureCard 
            title="Data-Driven Insights" 
            description="Comprehensive reporting and analytics to identify bottlenecks and optimize processes."
          />
          <FeatureCard 
            title="Mobile Accessibility" 
            description="Access from any device, enabling on-site submissions and approvals."
          />
          <FeatureCard 
            title="Secure Documentation" 
            description="All documents securely stored with proper versioning and access controls."
          />
          <FeatureCard 
            title="Scalability" 
            description="System can handle multiple projects simultaneously and adapt to projects of any size."
          />
        </div>
      </Section>

      {/* Implementation */}
      <Section title="Implementation Approach" id="implementation" bgColor="bg-gray-50">
        <p className="text-lg mb-6">
          Our implementation approach focuses on minimal disruption to ongoing projects while ensuring a smooth transition to the new system:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div>
            <h3 className="text-xl font-semibold mb-4">Phase 1: Setup & Configuration</h3>
            <ul className="list-disc pl-5 space-y-2">
              <li>Initial system setup and configuration</li>
              <li>User account creation and role assignment</li>
              <li>Activity list development with consultant input</li>
              <li>Integration with existing systems (if required)</li>
              <li>Security configuration and access control setup</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4">Phase 2: Training & Pilot</h3>
            <ul className="list-disc pl-5 space-y-2">
              <li>Role-specific training sessions for all users</li>
              <li>Comprehensive documentation and help resources</li>
              <li>Pilot implementation on a single project section</li>
              <li>Feedback collection and system refinement</li>
              <li>Process optimization based on initial usage</li>
            </ul>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-4">Phase 3: Full Deployment</h3>
            <ul className="list-disc pl-5 space-y-2">
              <li>System rollout across all project areas</li>
              <li>Migration of existing in-progress documents</li>
              <li>Ongoing technical support and troubleshooting</li>
              <li>Regular check-ins and progress monitoring</li>
              <li>Performance optimization as usage scales</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4">Phase 4: Optimization & Expansion</h3>
            <ul className="list-disc pl-5 space-y-2">
              <li>Collection of usage metrics and performance data</li>
              <li>System refinements based on user feedback</li>
              <li>Implementation of additional features as needed</li>
              <li>Development of custom reports and dashboards</li>
              <li>Preparation for future project implementations</li>
            </ul>
          </div>
        </div>
      </Section>

      {/* Conclusion */}
      <Section title="Conclusion" id="conclusion" bgColor="bg-white">
        <>
          <p className="text-lg mb-6">
            The Construction Document Approval Tracking System represents a significant advancement in how construction projects are managed and documented. By implementing this system, your organization will benefit from streamlined workflows, reduced administrative burden, and improved project visibility. The updated proposal ensures compliance with your requirements, including discipline-based IR serials, attachment support, multi-stage RFC/Time Schedule approval, Code B closure for minor comments, and Pouring Requests with multi-department approval.
          </p>
          <p className="text-lg mb-6">
            Our team is committed to delivering a solution that meets your specific needs and integrates seamlessly with your existing processes. We believe this system will not only improve efficiency in your current projects but will establish a new standard for document management in all future endeavors.
          </p>
          <p className="text-lg mb-6">
            We look forward to the opportunity to partner with you on this implementation and to demonstrate the substantial value this system will bring to your organization.
          </p>
        </>
      </Section>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center">© 2025 Construction Approval System Proposal</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
