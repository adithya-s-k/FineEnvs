"""Display names for the mock enterprise systems a General environment serves over MCP."""

ACRONYMS = set("""oa nc sap hr kb cmb crm hcm irm drg erp bi grc ap gl fa crcc ukg ibm nhs epa itsm hse kpmg fx cms
hrsd omv wms mes etq esr his esg bip mkuh epm lims qms rrb sas picc cdi rpa rsa pm jd gpo usps iq eas sis qa wjx ehs
cpic fis ar csps cems sgcc alm nyl hkex bacp elft capa mri pi it lms dr api hmrc irs sec fda cdc who ehr emr pos
kpi okr sla sop ppe vat gst cfo cto ceo rfp rfq po ai ml""".split())
BRANDS = {"dealcloud": "DealCloud", "sharepoint": "SharePoint", "servicenow": "ServiceNow", "netsuite": "NetSuite",
          "peoplesoft": "PeopleSoft", "successfactors": "SuccessFactors", "docusign": "DocuSign", "onesource": "ONESOURCE",
          "workiva": "Workiva", "quickbooks": "QuickBooks", "hubspot": "HubSpot", "powerbi": "Power BI", "sap": "SAP",
          "jira": "Jira", "github": "GitHub", "gitlab": "GitLab", "linkedin": "LinkedIn", "youtube": "YouTube"}


def pretty_system(name: str) -> str:
    def word(w: str) -> str:
        if w in BRANDS:
            return BRANDS[w]
        if w in ACRONYMS or any(c.isdigit() for c in w):
            return w.upper()
        return w.capitalize()
    return " ".join(word(w) for w in name.split("_") if w)
