//
//  AccountStatusView.swift
//  Tridify
//
//  Created by Maged Alosali on 30/04/2023.
//

import SwiftUI

struct AccountStatusView<TargetView: View>: View {
    
    private let questionText: String
    private let navigationText: String
    private let targetView : () -> TargetView
    
    var body: some View {
        HStack {
            Text (questionText)
            NavigationLink {
                targetView()
            } label: {
                Text(navigationText)
                    .foregroundColor(.link)
            }
        }
        .font(.headline)
        .fontWeight(.regular)
    }
    
    init (questionText: String, navigationText: String, targetView: @escaping () -> TargetView){
        self.questionText = questionText
        self.navigationText = navigationText
        self.targetView = targetView
    }
}

struct AccountStatusView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            AccountStatusView(questionText: "Already have an account?", navigationText: "Log in", targetView: {
                Text("Hello")
            })
        }
    }
}
