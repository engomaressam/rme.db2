# Construction Approval Tracking System: Features & Workflows

This document outlines the proposed features and workflows for the construction approval tracking system.

## 1. Core Concepts

*   **Project-Based:** The system organizes all documents and activities within specific construction projects.
*   **Role-Based Access Control:** Functionality is tailored to three main user roles: Contractor, Consultant, and Owner.
*   **Document Lifecycle Management:** Tracks each of the 9 specified document types from submission through review, revision, and final approval/closure.
*   **Transparency:** Provides visibility into document status and approval processes for relevant stakeholders.

## 2. User Roles & Permissions

*   **Contractor:**
    *   Submits: Inspection Requests (IR), Shop Drawings (SD), Material Submittals, Material Inspection Requests (MIR), Invoices, Requests for Information (RFI), Requests for Change (RFC), Time Schedules.
    *   Selects activities from a predefined list for IRs.
    *   Requests additions to the activity list.
    *   Attaches relevant files (PDF, DWG, etc.).
    *   Views the status of all submitted documents.
    *   Responds to Consultant notes/rejections (initiating revisions).
    *   Addresses and marks Non-Conformance Reports (NCRs) as ready for closure review.
    *   Views project activity list.
*   **Consultant:**
    *   Manages the master activity list for IRs (add, edit, approve Contractor requests).
    *   Reviews submitted documents (IR, SD, Material Submittal, MIR, Invoice, RFI, RFC, Schedule).
    *   Assigns approval codes (A, B, C, D as applicable per document type).
    *   Adds notes/comments, especially for B/C codes.
    *   Initiates NCRs.
    *   Reviews Contractor fixes for NCRs and marks them as Closed.
    *   Approves Invoices before they are visible/sent to the Owner.
    *   Answers RFIs.
    *   Approves/Rejects RFCs and Schedules.
    *   Views system-wide reports and project dashboards.
    *   (Potentially) Manages users within Contractor/Consultant organizations.
*   **Owner/Client:**
    *   View-only access to all documents, their statuses, attachments, and history across projects.
    *   View reports and dashboards.
    *   Receives approved invoices (or notifications).
*   **(Potential) System Administrator:**
    *   Overall system configuration.
    *   User management across all organizations.
    *   Project setup and archival.

## 3. Key System Modules & Features

*   **Project Management:**
    *   Create, view, and manage multiple construction projects.
    *   Assign users (Contractor, Consultant, Owner teams) to projects.
*   **Activity List Management (for IRs):**
    *   Consultant interface to define/upload/edit the activity list per project.
    *   Workflow for Contractor to request new activities and Consultant to approve/reject.
*   **Document Tracking (9 Modules):**
    *   **Common Features:**
        *   Unique Document ID & Revision Numbering (e.g., IR-001-Rev.00, SD-102-Rev.01).
        *   Clear Status Tracking (Submitted, Under Review, Approved (A), Approved with Notes (B), Rejected (C), Not Required (D), Open, Closed, Answered, etc.).
        *   Revision History Log (tracking changes, submissions, reviews).
        *   File Attachment Support (various formats, version control desirable).
        *   Commenting/Notes Feature (linked to specific revisions/reviews).
        *   Timestamping of all major actions.
        *   Robust Search & Filtering (by project, doc type, status, date, keywords).
    *   **Specific Workflows Implemented:** As detailed in requirements gathering (IR activity link, NCR open/close, Invoice approval path, etc.).
*   **Dashboard:**
    *   Personalized overview for each user.
    *   Summary of pending actions (items to review, items to resubmit).
    *   Quick view of recent activity and document status changes.
    *   Highlighting of overdue items (if deadlines are tracked).
*   **Notifications:**
    *   In-app and/or email alerts for critical events (configurable).
    *   Examples: New submission awaiting review, document approved/rejected, notes added, NCR raised, RFI answered, overdue task reminder.
*   **Reporting:**
    *   Predefined reports (e.g., Document Status Summary, Approval Cycle Time Analysis, Overdue Items Report, NCR Log).
    *   Ability to filter reports by project, date range, document type.
    *   Potential for exporting reports (e.g., to CSV/Excel).

## 4. Platform & Technology

*   **Primary Platform:** Secure, web-based application.
*   **Accessibility:** Responsive design for usability on desktops, tablets, and smartphones.
*   **(Optional):** Potential for future mobile app development.
*   **(Optional):** Integration capabilities (e.g., via APIs) with other relevant software (ERP, Project Management tools) if required.
