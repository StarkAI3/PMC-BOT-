# Generate sample queries for bot testing
_Exported on 23/07/2025 at 14:10:41 GMT+5:30 from Cursor (1.2.4)_

---

**User**

go through the mentioned file and give me sample marathi and english queries  to test the bot and the ensure all its features like context maintaing, link providing, etc.

---

**Cursor**

Here are some sample queries in both English and Marathi that you can use to test your bot’s features, such as context maintenance, link providing, and language handling. These are based on the actual entries in your `pmc_data_normalized.jsonl` file.

---

### English Sample Queries

1. **"Show me the latest circular about linking Aadhaar card and PAN card."**
   - *Tests:* Retrieval, link to PDF, context if you follow up with "Who issued it?" or "When was it released?"

2. **"Is there any training scheduled on the I-Got portal?"**
   - *Tests:* Retrieval, link to PDF, context if you follow up with "Which department is responsible?"

3. **"What is the latest update from the Labour Welfare Department?"**
   - *Tests:* Department filter, context if you follow up with "Give me more details about the insurance for contract workers."

4. **"Give me the agenda for the Standing Committee meeting on 18th July 2025."**
   - *Tests:* Date-based search, link to agenda PDF.

5. **"Who was recently promoted in the Engineering Cadre?"**
   - *Tests:* Entity extraction, context if you follow up with "Show me the order document."

6. **"Show me all circulars related to the General Administration department."**
   - *Tests:* Department filter, context if you follow up with "List only those from July 2025."

---

### Marathi Sample Queries

1. **"आधार कार्ड आणि पॅन कार्ड लिंक करण्याबाबतचा सर्क्युलर दाखवा."**
   - *Tests:* Marathi understanding, retrieval, PDF link.

2. **"कामगार कल्याण विभागाचे नवीन परिपत्रक काय आहे?"**
   - *Tests:* Department filter, Marathi, context if you follow up with "त्यात विमा संदर्भात काय आहे?"

3. **"पुणे मनपा अधिकारी/सेवकांच्या पाल्यांचा गुणगौरव समारंभासाठी अर्ज मागविण्याबाबत माहिती द्या."**
   - *Tests:* Retrieval, context, PDF link.

4. **"Standing Committee च्या १८ जुलै २०२५ च्या बैठकीचे अजेंडा द्या."**
   - *Tests:* Date-based search, Marathi, PDF link.

5. **"महानगरपालिकेच्या वेबसाईट प्रशिक्षणाबद्दल माहिती द्या."**
   - *Tests:* Topic-based search, context, PDF link.

---

### Context Maintenance Test (Follow-up Example)

- **User:** "Show me the latest circular about linking Aadhaar card and PAN card."
- **User (follow-up):** "Who issued it?"
- **User (follow-up):** "Give me the download link."

---

### Link Providing Test

- **User:** "Give me the PDF for the SAP Sasa-40 Training Workshop."
- **Expected:** Bot provides the direct PDF link.

---

### Language Switching Test

- **User:** "Show me the latest circular about linking Aadhaar card and PAN card."
- **User (follow-up, in Marathi):** "हे कोणत्या विभागाने जारी केले आहे?"

---

These queries will help you comprehensively test the bot’s retrieval, context, language, and link-providing features. If you need more or want them tailored to specific features, let me know!

