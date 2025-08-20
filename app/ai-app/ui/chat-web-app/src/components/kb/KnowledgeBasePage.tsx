import KBPanel from "./KBPanel.tsx";

function KnowledgeBasePage({onClose}: { onClose?: () => void }) {
    return (<div className="fixed inset-0 z-50 flex">
        <div className="absolute inset-0 bg-transparent" onClick={onClose}/>
        <KBPanel onClose={onClose}/>
    </div>)
}

export default KnowledgeBasePage