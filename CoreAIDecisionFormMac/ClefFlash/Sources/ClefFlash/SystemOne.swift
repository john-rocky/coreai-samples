// Vendored from github.com/john-rocky/coreai-model-zoo apps/ClefFlash/Sources/ClefFlash/SystemOne.swift at 5ef2247, unmodified below this line.
// SystemOne — the request and the response, in the shape of the checkpoint's `systemone()` (joint_schema_model.py at
// the pinned revision; conversion/clef_flash/host.py `validate_request`, decide.py `systemone_answer` / `response`):
//
//   request  {"model": str, "state": any, "questions": {question_id: {"type": "noul" | "choice" | "score",
//            "instructions"?: any, "criteria"?: {option_id: description} (choice; noul overrides true / false) |
//            [level description, ...] (score)}}}            JSON order kept (questions are answered in this order)
//   response {"model", "answers": {question_id: answer}, "usage": {"input_tokens": T, "output_tokens": 0}}
//            noul   {"type", "noul": p(true)}
//            choice {"type", "choice": the first most probable option id in the request's order, "confidence": its p,
//                    "probabilities": {option_id: p} in the request's order}
//            score  {"type", "score": sum(level * p), "confidence": max p, "legend": {level: criterion},
//                    "probabilities": {level: p}}
//   every number rounded with Python's round(x, 4); the probabilities are the float32 softmax values as doubles

import Foundation

@available(macOS 27, *)
public struct SystemOneRequest: Sendable {
    public struct Question: Sendable {
        public let id: String
        /// "noul", "choice" or "score"
        public let type: String
        /// nil when absent; the prompt uses the id when this is nil, null or ""
        public let instructions: JSONValue?
        public let criteria: JSONValue?
    }

    public static let questionTypes: [String: Int] = ["noul": 0, "choice": 1, "score": 2]
    public static let noulCriteria: [JSONMember] = [
        JSONMember("true", .string("The proposition is true or the answer is yes.")),
        JSONMember("false", .string("The proposition is false or the answer is no.")),
    ]

    public let model: String
    public let state: JSONValue
    public let questions: [Question]
    /// The request as given.
    public let json: JSONValue

    public init(data: Data) throws { try self.init(json: try JSONParser.parse(data)) }

    /// The checks the author's `systemone()` makes before encoding (`host.validate_request`).
    public init(json: JSONValue) throws {
        guard case .object = json else { throw ClefFlashError.request("the request is not a JSON object") }
        guard case .string(let model)? = json["model"], let state = json["state"] else {
            throw ClefFlashError.request("model and state are required")
        }
        guard let qs = json["questions"]?.members, !qs.isEmpty else {
            throw ClefFlashError.request("at least one question is required")
        }
        var questions: [Question] = []
        for m in qs {
            guard case .object = m.value else { throw ClefFlashError.request("\(m.key): a question is an object") }
            guard case .string(let type)? = m.value["type"], Self.questionTypes[type] != nil else {
                throw ClefFlashError.request("\(m.key): type must be noul, choice, or score")
            }
            let criteria = m.value["criteria"]
            if type != "noul" && !(criteria?.isTruthy ?? false) {
                throw ClefFlashError.request("\(m.key): criteria must not be empty")
            }
            questions.append(Question(id: m.key, type: type, instructions: m.value["instructions"], criteria: criteria))
        }
        self.model = model
        self.state = state
        self.questions = questions
        self.json = json
    }
}

@available(macOS 27, *)
extension SystemOneRequest.Question {
    /// (option id, description) in the order the head scores them (the author's `question_options`): noul = true,
    /// false (the default criteria, overridable); choice = option ids sorted by code point; score = levels 0 ..< n.
    /// A nil description = JSON null (left out of the option's rendering).
    public func options() throws -> [(id: String, description: JSONValue?)] {
        func value(_ v: JSONValue) -> JSONValue? { v.isNull ? nil : v }
        switch type {
        case "noul":
            var c = SystemOneRequest.noulCriteria
            if let given = criteria, given.isTruthy {
                guard let m = given.members else { throw ClefFlashError.request("\(id): noul criteria must be an object") }
                for x in m {
                    if let k = c.firstIndex(where: { $0.key == x.key }) { c[k].value = x.value } else { c.append(x) }
                }
            }
            return ["true", "false"].map { k in (k, value(c.first(where: { $0.key == k })!.value)) }
        case "choice":
            guard let m = criteria?.members else { throw ClefFlashError.request("\(id): choice criteria must be an object") }
            return m.sorted { PythonJSON.codePointLess($0.key, $1.key) }.map { ($0.key, value($0.value)) }
        default:
            if let a = criteria?.array { return a.enumerated().map { (String($0.offset), value($0.element)) } }
            if let m = criteria?.members { return m.enumerated().map { (String($0.offset), .string($0.element.key)) } }
            throw ClefFlashError.request("\(id): score criteria must be a list")
        }
    }

    /// The criteria in the request's order as `systemone_answer` iterates them (choice: the option ids; score: the
    /// level descriptions).
    var requestOrder: [JSONValue] {
        if let m = criteria?.members { return m.map { .string($0.key) } }
        return criteria?.array ?? []
    }
}

// MARK: - Response

@available(macOS 27, *)
public enum SystemOneResponse {
    /// `systemone_answer` for one question. `probabilities` = (option id, p) in the head's option order.
    public static func answer(_ q: SystemOneRequest.Question, probabilities: [(id: String, p: Float)]) throws -> JSONValue {
        var p: [String: Double] = [:]
        for (id, v) in probabilities { p[id] = Double(v) }
        let needed: [String]
        switch q.type {
        case "noul": needed = ["true"]
        case "choice": needed = q.requestOrder.compactMap(\.string)
        default: needed = (0..<q.requestOrder.count).map(String.init)
        }
        for id in needed where p[id] == nil {
            throw ClefFlashError.request("\(q.id): no probability for option \(id)")
        }
        let r = { (x: Double) -> JSONValue in .double(PythonJSON.pyRound(x, 4)) }
        switch q.type {
        case "noul":
            return .obj([("type", .string("noul")), ("noul", r(p["true"]!))])
        case "choice":
            let options = needed
            var best = options[0]
            for o in options.dropFirst() where p[o]! > p[best]! { best = o }
            return .obj([("type", .string("choice")), ("choice", .string(best)), ("confidence", r(p[best]!)),
                         ("probabilities", .obj(options.map { ($0, r(p[$0]!)) }))])
        default:
            let criteria = q.requestOrder
            let levels = needed
            var score = 0.0                       // Python 3.11 sum(): left to right from int 0
            for (k, l) in levels.enumerated() { score += Double(k) * p[l]! }
            var conf = p[levels[0]]!
            for l in levels.dropFirst() where p[l]! > conf { conf = p[l]! }
            return .obj([("type", .string("score")), ("score", r(score)), ("confidence", r(conf)),
                         ("legend", .obj(zip(levels, criteria).map { ($0, $1) })),
                         ("probabilities", .obj(levels.map { ($0, r(p[$0]!)) }))])
        }
    }

    /// The whole response: answers in the request's question order.
    public static func make(_ request: SystemOneRequest, probabilities: [[(id: String, p: Float)]], tokens: Int) throws -> JSONValue {
        guard probabilities.count == request.questions.count else {
            throw ClefFlashError.request("\(probabilities.count) probability rows for \(request.questions.count) questions")
        }
        let answers = try zip(request.questions, probabilities).map { q, p in (q.id, try answer(q, probabilities: p)) }
        return .obj([("model", .string(request.model)), ("answers", .obj(answers)),
                     ("usage", .obj([("input_tokens", .int(tokens)), ("output_tokens", .int(0))]))])
    }
}
